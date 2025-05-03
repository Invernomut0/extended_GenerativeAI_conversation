"""The Generative AI Conversation integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from proto.marshal.collections import maps, repeated
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, FunctionDeclaration
import yaml
import voluptuous as vol

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


def _recursive_proto_to_dict(proto_obj: Any) -> Any:
    """Recursively convert protobuf messages/composites to dict/list."""
    # Check for specific protobuf composite types first
    if isinstance(proto_obj, maps.MapComposite):
        return {
            key: _recursive_proto_to_dict(value) for key, value in proto_obj.items()
        }
    if isinstance(proto_obj, repeated.RepeatedComposite):
        return [_recursive_proto_to_dict(item) for item in proto_obj]

    # Duck-typing check for Part-like objects (has text or function_call)
    # This replaces the explicit 'isinstance(proto_obj, Part)' check
    if (
        hasattr(proto_obj, "text")
        or hasattr(proto_obj, "function_call")
        or hasattr(proto_obj, "function_response")
    ):
        part_dict = {}
        if hasattr(proto_obj, "text") and proto_obj.text:
            part_dict["text"] = proto_obj.text
        if hasattr(proto_obj, "function_call") and proto_obj.function_call:
            part_dict["function_call"] = {
                "name": proto_obj.function_call.name,
                "args": _recursive_proto_to_dict(proto_obj.function_call.args),
            }
        if hasattr(proto_obj, "function_response") and proto_obj.function_response:
            part_dict["function_response"] = _recursive_proto_to_dict(
                proto_obj.function_response
            )
        # Only return the dict if it's not empty, otherwise fall through
        if part_dict:
            return part_dict

    # Generic handling for other protobuf message types
    if hasattr(proto_obj, "_meta") and hasattr(proto_obj, "_pb"):
        msg_dict = {}
        for field_descriptor in proto_obj._meta.fields:
            field_name = field_descriptor.name
            if hasattr(proto_obj, field_name):
                value = getattr(proto_obj, field_name)
                if value or isinstance(value, (bool, int, float)):
                    msg_dict[field_name] = _recursive_proto_to_dict(value)
        # Return the dict if not empty, otherwise fall through
        if msg_dict:
            return msg_dict

    # Assume it's a basic serializable type
    return proto_obj


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

        # Configure the Generative AI API with all necessary parameters
        config = {"api_key": entry.data[CONF_API_KEY]}
        if base_url := entry.data.get(CONF_BASE_URL):
            config["client_options"] = {"api_endpoint": base_url}
        if api_version := entry.data.get(CONF_API_VERSION):
            config["api_version"] = api_version

        genai.configure(**config)

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

        # Convert the raw response object to a serializable dictionary
        try:
            serializable_response = _recursive_proto_to_dict(query_response.response)
        except Exception as e:
            _LOGGER.warning("Failed to serialize API response for event: %s", e)
            # Fallback to a simple representation or None if serialization fails
            serializable_response = {
                "error": "Failed to serialize response",
                "details": str(e),
            }
            # Or potentially just log the raw response text if available
            if hasattr(query_response.response, "text"):
                serializable_response = {"text": query_response.response.text}

        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "response": serializable_response,  # Use the serialized version
                "user_input": user_input,  # user_input should be serializable
                "messages": messages,  # messages is already a list of dicts
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

        # Convert the messages to the format expected by Gemini API
        gemini_messages = []
        for message in messages:
            role = message["role"]
            if role == "system":
                # Gemini doesn't have system messages, prepend to first user message
                continue
            elif role == "user":
                # Combine system prompt with first user message if necessary
                content = message.get("content")
                if not content:
                    _LOGGER.warning("User message missing content: %s", message)
                    continue

                if len(gemini_messages) == 0 and messages[0]["role"] == "system":
                    system_content = messages[0].get("content", "")
                    gemini_messages.append(
                        {
                            "role": "user",
                            "parts": [system_content + "\n\n" + content],
                        }
                    )
                else:
                    gemini_messages.append({"role": "user", "parts": [content]})

            elif role == "assistant":
                # Assistant message maps to model role in history
                # Handle both simple content and parts list (for function calls)
                if "parts" in message:
                    # If parts exist (likely from a function call turn), use them directly
                    gemini_messages.append({"role": "model", "parts": message["parts"]})
                elif "content" in message:
                    # Otherwise, use the content
                    gemini_messages.append(
                        {"role": "model", "parts": [message["content"]]}
                    )
                else:
                    _LOGGER.warning(
                        "Assistant message missing parts or content: %s", message
                    )

            elif role == "tool":
                # Our internal 'tool' role (function result) maps to Gemini's 'function' role for the response turn
                if (
                    "parts" in message
                    and message["parts"]
                    and "function_response" in message["parts"][0]
                ):
                    # Use role: "function" when sending function results back to the API
                    gemini_messages.append(
                        {"role": "function", "parts": message["parts"]}
                    )
                else:
                    _LOGGER.warning(
                        "Tool message missing or has invalid 'parts' structure: %s",
                        message,
                    )

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
        # History should now include user, model (w/ function call), function (w/ result) roles correctly
        chat = model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
        _LOGGER.debug(
            "Chat history sent to start_chat: %s",
            gemini_messages[:-1] if gemini_messages else "[]",
        )

        # Generate the response
        last_message_content = gemini_messages[-1]["parts"] if gemini_messages else []
        last_message_role = (
            gemini_messages[-1]["role"] if gemini_messages else "user"
        )  # Get the role of the last message
        _LOGGER.debug(
            "Last message being sent: role=%s, content=%s",
            last_message_role,
            last_message_content,
        )

        # The lambda sends the last message (which could be user text OR the function response)
        if function_declarations and function_calling_enabled:
            response = await self.hass.async_add_executor_job(
                lambda: chat.send_message(
                    last_message_content,
                    generation_config=generation_config,
                    tools=function_declarations,
                )
            )
        else:
            response = await self.hass.async_add_executor_job(
                lambda: chat.send_message(
                    last_message_content,
                    generation_config=generation_config,
                )
            )

        # --- Start: Cleaned Token Counting Block ---
        prompt_tokens = 0
        candidates_tokens = 0
        total_tokens = 0

        _LOGGER.debug(
            "Checking for usage_metadata on response object (type: %s)", type(response)
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            usage_metadata = response.usage_metadata
            _LOGGER.debug("Found usage_metadata: %s", usage_metadata)
            try:
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                candidates_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(usage_metadata, "total_token_count", 0)

                # Ensure types are integers
                prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
                candidates_tokens = (
                    int(candidates_tokens) if candidates_tokens is not None else 0
                )
                total_tokens = int(total_tokens) if total_tokens is not None else 0

                _LOGGER.debug(
                    "Gemini token usage: prompt=%d, candidates=%d, total=%d",
                    prompt_tokens,
                    candidates_tokens,
                    total_tokens,
                )
            except (AttributeError, TypeError, ValueError) as e:
                _LOGGER.error(
                    "Error accessing token counts within usage_metadata: %s",
                    e,
                    exc_info=True,
                )
                # Reset tokens on error
                prompt_tokens = 0
                candidates_tokens = 0
                total_tokens = 0
        else:
            _LOGGER.warning(
                "Usage metadata not found or is None in response object. Response: %s",
                response,
            )

        # Defensive check: Ensure total_tokens is an integer before comparison
        if not isinstance(total_tokens, int):
            _LOGGER.error(
                "total_tokens is not an integer after processing: %s (%s)",
                total_tokens,
                type(total_tokens),
            )
            total_tokens = 0  # Defaulting to 0

        if total_tokens > context_threshold:
            _LOGGER.debug(
                "Token usage (%d) exceeds threshold (%d), truncating history",
                total_tokens,
                context_threshold,
            )
            await self.truncate_message_history(messages, exposed_entities, user_input)
        # --- End: Cleaned Token Counting Block ---

        # Handle function call if present
        function_call = None
        response_text_parts = []  # Collect text parts
        response_text = None  # Initialize response_text

        try:
            if response.candidates and response.candidates[0].content.parts:
                _LOGGER.debug(
                    "Iterating through %d response parts",
                    len(response.candidates[0].content.parts),
                )
                # Iterate through all parts in the first candidate's content
                for i, part in enumerate(response.candidates[0].content.parts):
                    _LOGGER.debug("Processing part %d: %s", i, part)  # Log the raw part

                    # Check for text first and append it
                    if hasattr(part, "text") and part.text:
                        _LOGGER.debug("Part %d has text: %s", i, part.text)
                        response_text_parts.append(part.text)

                    # Then check for function call
                    if hasattr(part, "function_call") and part.function_call:
                        # Log the found function_call object before validation
                        _LOGGER.debug(
                            "Part %d has function_call object: %s",
                            i,
                            part.function_call,
                        )
                        # Ensure function_call is valid before assigning and breaking
                        if (
                            hasattr(part.function_call, "name")
                            and part.function_call.name
                        ):
                            function_call = part.function_call
                            _LOGGER.debug(
                                "Part %d has VALID function_call: %s",  # Clarified log
                                i,
                                function_call.name,
                            )
                            # Prioritize function call; stop processing parts if found
                            break
                        else:
                            # Log the specific reason why it was considered invalid
                            _LOGGER.warning(
                                "Part %d has function_call attribute but it's invalid (e.g., missing name). function_call object: %s",
                                i,
                                part.function_call,  # Log the object again for clarity
                            )
                    # Optional: Log if the part doesn't have function_call attribute at all
                    # else:
                    #    _LOGGER.debug("Part %d does not have function_call attribute", i)

            # Log state after loop
            _LOGGER.debug("After loop: function_call is %s", function_call)
            _LOGGER.debug("After loop: response_text_parts is %s", response_text_parts)

            # --- Start: Modified Logic ---
            # If a function call was found, store model response and execute
            if function_call:
                _LOGGER.info("Proceeding with function call: %s", function_call.name)

                # Store the raw model response parts (containing the function call)
                # in the internal history before executing the function.
                if response.candidates and response.candidates[0].content.parts:
                    model_raw_parts = response.candidates[0].content.parts
                    # Convert raw parts to serializable dict/list structure
                    serializable_parts = _recursive_proto_to_dict(model_raw_parts)
                    messages.append({"role": "assistant", "parts": serializable_parts})
                    _LOGGER.debug(
                        "Appending assistant message (serializable parts) before function call: %s",
                        messages[-1],
                    )
                else:
                    _LOGGER.warning(
                        "Could not retrieve model parts to store before function call"
                    )

                return await self.execute_function_call(
                    user_input,
                    messages,  # Pass updated messages list
                    function_call,
                    exposed_entities,
                    n_requests + 1,
                )

            # If NO function call was found, combine collected text parts
            _LOGGER.debug("No function call found, processing text parts")
            if response_text_parts:
                response_text = "\n".join(
                    response_text_parts
                ).strip()  # Join parts with newline
                _LOGGER.debug("Combined response_text: %s", response_text)
            # else: response_text remains None

            # --- End: Modified Logic ---

        except ValueError as e:
            _LOGGER.error("ValueError processing response parts: %s", e)
            raise HomeAssistantError(
                "Failed to process Generative AI response content"
            ) from e
        except Exception as e:
            _LOGGER.warning("Error processing response parts: %s", e, exc_info=True)
            raise HomeAssistantError(
                "Unexpected error processing response parts"
            ) from e

        # If we reach here, it means NO function call was found.
        # Process the text response (if any).
        if response_text is not None:
            _LOGGER.info("Processed response text: %s", response_text)
            # Create a response message for the assistant
            message = {
                "role": "assistant",
                "content": response_text,
            }
            _LOGGER.debug("Returning text response message: %s", message)
            return GenerativeAIQueryResponse(response=response, message=message)
        else:
            # This path is hit only if NO function_call AND NO text was found.
            _LOGGER.error(
                "No usable text or function call found after processing parts. Raw content: %s",
                response.candidates[0].content
                if response.candidates
                else "No candidates",
            )
            raise HomeAssistantError("No content found in Generative AI response")

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        function_call,
        exposed_entities,
        n_requests,
    ) -> GenerativeAIQueryResponse:
        function_name = function_call.name
        _LOGGER.debug("Executing function call: %s", function_name)
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return await self.execute_function(
                user_input,
                messages,
                function_call,
                exposed_entities,
                n_requests,
                function,
            )
        _LOGGER.error("Function spec not found for '%s'", function_name)
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        function_call,
        exposed_entities,
        n_requests,
        function,
    ) -> GenerativeAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            # Use the recursive helper for deep conversion
            arguments = _recursive_proto_to_dict(function_call.args)
            _LOGGER.debug("Function arguments (deep converted): %s", arguments)
        except Exception as err:
            _LOGGER.error(
                "Failed to parse/convert arguments from function call: %s",
                function_call,
                exc_info=True,
            )
            raise ParseArgumentsFailed(str(function_call)) from err

        try:
            result = await function_executor.execute(
                self.hass, function["function"], arguments, user_input, exposed_entities
            )
            _LOGGER.debug("Function result: %s", result)
            serializable_result = (
                str(result) if result is not None else "Function executed successfully."
            )
        # Catch specific voluptuous validation errors from service calls
        except vol.MultipleInvalid as e:
            _LOGGER.error(
                "Validation error executing function %s: %s", function_call.name, e
            )
            # Provide a more specific message if it's likely a config issue
            if "extra keys not allowed" in str(e) or "required key not provided" in str(
                e
            ):
                serializable_result = (
                    f"Error executing function {function_call.name}: Service call failed validation. "
                    f"Check the service call data in the function's YAML definition. Error: {e}"
                )
            else:
                serializable_result = (
                    f"Error executing function {function_call.name}: {e}"
                )
        except Exception as e:
            _LOGGER.error(
                "Error executing function %s: %s", function_call.name, e, exc_info=True
            )
            serializable_result = f"Error executing function {function_call.name}: {e}"

        # Append the result message with the 'tool' role for internal tracking
        messages.append(
            {
                "role": "tool",  # Internal role remains 'tool'
                "parts": [
                    {
                        "function_response": {
                            "name": function_call.name,
                            "response": {"content": serializable_result},
                        }
                    }
                ],
            }
        )
        _LOGGER.debug("Appending tool message to internal history: %s", messages[-1])
        # Call query again, the loop in query will handle the 'tool' -> 'function' role conversion for the API call
        return await self.query(user_input, messages, exposed_entities, n_requests)


class GenerativeAIQueryResponse:
    """Generative AI query response value object."""

    def __init__(self, response, message) -> None:
        """Initialize Generative AI query response value object."""
        self.response = response
        self.message = message
