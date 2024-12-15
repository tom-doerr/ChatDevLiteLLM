import pytest
from unittest.mock import MagicMock, patch

from camel.model_backend import ModelFactory, ModelType
from camel.messages import OpenAIMessage

@patch("camel.model_backend.litellm.completion")
def test_deepseek_model_does_not_pass_proxies_arg(mock_completion):
    # Arrange
    model_factory = ModelFactory()
    model = model_factory.create(ModelType.DEEPSEEK, {})
    messages = [OpenAIMessage(role="user", content="test message")]
    kwargs = {"messages": messages, "proxies": {"http": "test_proxy"}}

    # Act
    model.run(**kwargs)

    # Assert
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert "proxies" not in call_kwargs
    
    
@patch("camel.model_backend.litellm.completion")
def test_deepseek_model_logit_bias_arg_removed(mock_completion):
    # Arrange
    model_factory = ModelFactory()
    model = model_factory.create(ModelType.DEEPSEEK, {"logit_bias": {1: 100}})
    messages = [OpenAIMessage(role="user", content="test message")]
    kwargs = {"messages": messages}

    # Act
    model.run(**kwargs)

    # Assert
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args[1]
    assert "logit_bias" not in call_kwargs
