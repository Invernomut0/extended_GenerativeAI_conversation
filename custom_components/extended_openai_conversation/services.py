import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import google.generativeai as genai
import voluptuous as vol

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, SERVICE_QUERY_IMAGE

QUERY_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        vol.Required("model", default="gemini-pro-vision"): cv.string,
        vol.Required("prompt"): cv.string,
        vol.Required("images"): vol.All(cv.ensure_list, [{"url": cv.string}]),
        vol.Optional("max_tokens", default=300): cv.positive_int,
    }
)

_LOGGER = logging.getLogger(__package__)


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the extended generativeai conversation component."""

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        try:
            model_name = call.data["model"]
            api_key = hass.data[DOMAIN][call.data["config_entry"]]["api_key"]
            
            # Configure the Generative AI with the API key
            genai.configure(api_key=api_key)
            
            # Get the model
            model = genai.GenerativeModel(model_name)
            
            # Process images
            image_parts = []
            for image in call.data["images"]:
                image_data = to_image_data(hass, image)
                if "base64" in image_data:
                    # Image is a base64 encoded local file
                    image_parts.append(
                        genai.types.Blob(
                            mime_type=image_data["mime_type"],
                            data=base64.b64decode(image_data["base64"])
                        )
                    )
                else:
                    # Image is a URL
                    image_parts.append(image_data["url"])
            
            # Create the parts list with the prompt and images
            parts = [call.data["prompt"]] + image_parts
            
            # Generate the response
            response = await hass.async_add_executor_job(
                model.generate_content,
                parts,
                genai.types.GenerationConfig(max_output_tokens=call.data["max_tokens"])
            )
            
            # Convert response to dictionary
            response_dict = {
                "text": response.text,
                "status": "complete",
                "model": model_name,
            }
            
            _LOGGER.info("Response from Generative AI: %s", response_dict)
            
        except Exception as err:
            raise HomeAssistantError(f"Error generating image response: {err}") from err

        return response_dict

    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_IMAGE,
        query_image,
        schema=QUERY_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def to_image_data(hass: HomeAssistant, image):
    """Convert url to base64 encoded image if local."""
    url = image["url"]

    if urlparse(url).scheme in cv.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST:
        return {"url": url}

    if not hass.config.is_allowed_path(url):
        raise HomeAssistantError(
            f"Cannot read `{url}`, no access to path; "
            "`allowlist_external_dirs` may need to be adjusted in "
            "`configuration.yaml`"
        )
    if not Path(url).exists():
        raise HomeAssistantError(f"`{url}` does not exist")
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type is None or not mime_type.startswith("image"):
        raise HomeAssistantError(f"`{url}` is not an image")

    return {"mime_type": mime_type, "base64": encode_image(url)}


def encode_image(image_path):
    """Convert to base64 encoded image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
