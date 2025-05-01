"""The Generative AI Conversation integration."""
from __future__ import annotations

import json
import logging
from typing import Literal

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, FunctionDeclaration
import yaml

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import (
    get_function_executor,
    validate_authentication,
)
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


# hass.data key for agent.
DATA_AGENT = "agent"


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Generative AI Conversation."""
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Generative AI Conversation from a config entry."""

    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except Exception as err:
        _LOGGER.error("Authentication error: %s", err)
        return False
    except Exception as err:
        raise ConfigEntryNotReady(err) from err

    agent = GenerativeAIAgent(hass, entry)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Generative AI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class GenerativeAIAgent(conversation.AbstractConversationAgent):
    """Generative AI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}
        self.model = None
        
        # Configure the Generative AI API
        genai.configure(api_key=entry.data[CONF_API_KEY])

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        exposed_entities = self.get_exposed_entities()

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            try:
                system_message = self._generate_system_message(
                    exposed_entities, user_input
                )
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [system_message]
        user_message = {"role": "user", "content": user_input.text}
        if self.entry.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME):
            user = user_input.context.user_id
            if user is not None:
                user_message[ATTR_NAME] = user

        messages.append(user_message)

        try:
            query_response = await self.query(user_input, messages, exposed_entities, 0)
        except Exception as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to Generative AI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append(query_response.message)
        self.history[conversation_id] = messages

        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "response": query_response.response,
                "user_input": user_input,
                "messages": messages,
            },
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(query_response.message.get("content", ""))
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ):
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        prompt = self._async_generate_prompt(raw_prompt, exposed_entities, user_input)
        return {"role": "system", "content": prompt}

    def _async_generate_prompt(
        self,
        raw_prompt: str,
        exposed_entities,
        user_input: conversation.ConversationInput,
    ) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    def get_functions(self):
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def truncate_message_history(
        self, messages, exposed_entities, user_input: conversation.ConversationInput
    ):
        """Truncate message history."""
        strategy = self.entry.options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            last_user_message_index = None
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    last_user_message_index = i
                    break

            if last_user_message_index is not None:
                del messages[1:last_user_message_index]
                # refresh system prompt when all messages are deleted
                messages[0] = self._generate_system_message(
                    exposed_entities, user_input
                )

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ) -> GenerativeAIQueryResponse:
        """Process a sentence."""
        model_name = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        use_tools = self.entry.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
        context_threshold = self.entry.options.get(
            CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
        )
        functions = self.get_functions()
        function_calling_enabled = True
        if n_requests == self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ):
            function_calling_enabled = False

        _LOGGER.info("Prompt for %s: %s", model_name, json.dumps(messages))

        # Get the Gemini model
        model = genai.GenerativeModel(model_name=model_name)
        
        # Convert the messages to the format expected by Gemini
        gemini_messages = []
        for message in messages:
            role = message["role"]
            if role == "system":
                # Gemini doesn't have system messages, prepend to first user message
                continue
            elif role == "user":
                if len(gemini_messages) == 0 and messages[0]["role"] == "system":
                    # Add system message content to first user message
                    gemini_messages.append({
                        "role": "user",
                        "parts": [messages[0]["content"] + "\n\n" + message["content"]]
                    })
                else:
                    gemini_messages.append({
                        "role": "user",
                        "parts": [message["content"]]
                    })
            elif role == "function" or role == "tool":
                # Function response becomes assistant message in Gemini
                gemini_messages.append({
                    "role": "model",
                    "parts": [message["content"]]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [message["content"]]
                })

        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        # Prepare function declarations if needed
        function_declarations = []
        if functions and function_calling_enabled and use_tools:
            for function in functions:
                function_declarations.append(
                    FunctionDeclaration(
                        name=function["spec"]["name"],
                        description=function["spec"]["description"],
                        parameters=function["spec"]["parameters"],
                    )
                )

        # Create the chat session
        chat = model.start_chat(
            history=gemini_messages[:-1] if gemini_messages else []
        )

        # Generate the response
        if function_declarations and function_calling_enabled:
            response = await self.hass.async_add_executor_job(
                chat.send_message,
                gemini_messages[-1]["parts"][0],
                generation_config=generation_config,
                tools=[function_declarations],
            )
        else:
            response = await self.hass.async_add_executor_job(
                chat.send_message,
                gemini_messages[-1]["parts"][0],
                generation_config=generation_config,
            )

        _LOGGER.info("Response %s", response.text)

        # Check token usage and truncate if needed
        if hasattr(response, 'usage') and response.usage.total_tokens > context_threshold:
            await self.truncate_message_history(messages, exposed_entities, user_input)

        # Handle function call if present
        has_function_call = hasattr(response, 'candidates') and response.candidates[0].get('function_call')
        if has_function_call:
            return await self.execute_function_call(
                user_input, messages, response, exposed_entities, n_requests + 1
            )

        # Create a response message
        message = {
            "role": "assistant",
            "content": response.text,
        }

        return GenerativeAIQueryResponse(response=response, message=message)

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        response,
        exposed_entities,
        n_requests,
    ) -> GenerativeAIQueryResponse:
        function_call = response.candidates[0]['function_call']
        function_name = function_call['name']
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return await self.execute_function(
                user_input,
                messages,
                response,
                exposed_entities,
                n_requests,
                function,
                function_call
            )
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        response,
        exposed_entities,
        n_requests,
        function,
        function_call,
    ) -> GenerativeAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = function_call['args']
        except Exception as err:
            raise ParseArgumentsFailed(str(function_call)) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )

        messages.append(
            {
                "role": "function",
                "name": function_call['name'],
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)


class GenerativeAIQueryResponse:
    """Generative AI query response value object."""

    def __init__(
        self, response, message
    ) -> None:
        """Initialize Generative AI query response value object."""
        self.response = response
        self.message = message
