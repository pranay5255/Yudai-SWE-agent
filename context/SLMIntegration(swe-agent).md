# Systematic Plan: Adding SLM (Small Language Model) Functionality to SWE-agent

## Overview

This plan outlines the systematic approach to integrate task-specific Small Language Models (SLMs) into the SWE-agent, allowing the system to use specialized models for specific tasks while leveraging the existing `litellm`-based architecture. The goal is to provide a flexible mechanism for users to define, configure, and use SLMs for software engineering tasks.

## Detailed Agent Execution Analysis

To effectively integrate SLMs, it's crucial to understand the existing execution flow of `swe-agent`. The primary components are the configuration system, the model classes, the agent itself, and the environment it interacts with.

### 1. Core Components

**1.1 Configuration (`sweagent/utils/config.py`)**

- **Purpose**: Manages all configurations for the agent, environment, and models.
- **Mechanism**: It uses `simple_parsing` to define hierarchical configurations from YAML files and command-line arguments. The main configuration objects are defined across various files and composed together.
- **Integration Point**: This is where we will define the configurations for our SLMs, making them available to the system. We can add a new configuration section for a dictionary of SLM providers.

**1.2 Model Abstraction (`sweagent/agent/models.py`)**

- **Purpose**: Provides a unified interface for interacting with different language models.
- **Key Classes**:
    - `AbstractModel`: An abstract base class defining the common interface for all models, with the central method being `query`.
    - `GenericAPIModelConfig`: A Pydantic model that defines the configuration for any API-based model compatible with `litellm`. This is the key to our integration, as it can be used to configure SLMs served via an API endpoint.
    - `LiteLLMModel`: The concrete implementation of `AbstractModel` that uses the `litellm` library to send requests to the configured model. It handles API calls, retries, cost tracking, and error handling.
- **SLM Integration Point**: We can leverage `GenericAPIModelConfig` to define our SLMs. By setting the `api_base` to our SLM's endpoint, we can use `LiteLLMModel` to interact with it without any changes to the class itself.

**1.3 Agent Logic (`sweagent/agent/agents.py`)**

- **Purpose**: Orchestrates the interaction between the model and the environment to solve a given software engineering task.
- **Key Class**: `DefaultAgent` contains the main thought-action-observation loop.
- **Execution Flow**:
    1. The `run()` method initializes the environment and trajectory.
    2. It enters a loop that continues until the task is resolved or a limit is reached.
    3. Inside the loop, `handle_action()` is called.
    4. `handle_action()` builds the prompt history and calls `self.model.query(history)` to get the next thought and action from the language model.
    5. The returned action is parsed and executed in the `SWEEnv` environment.
    6. The observation from the environment is appended to the history, and the loop continues.
- **SLM Integration Point**: The agent is initialized with a single model instance. To use an SLM, we need to ensure that the `self.model` attribute of the agent is an instance of `LiteLLMModel` configured with the SLM's parameters. This selection should happen at the time of agent instantiation.

**1.4 Main Execution Script (`sweagent/run/run_single.py`)**

- **Purpose**: The entry point for running a single `swe-agent` task.
- **Flow**:
    1. Parses command-line arguments and loads configurations.
    2. Instantiates the `SWEEnv` environment.
    3. Instantiates the `DefaultAgent` with the configured model.
    4. Calls `agent.run()` to start the task.
- **SLM Integration Point**: This script is the ideal place to implement the logic for selecting the SLM. We can add a CLI argument to specify an SLM, look up its configuration, and use that to initialize the agent.

### 2. SLM Integration Strategy

The existing architecture is highly modular and flexible, which allows for a clean integration of SLMs. Instead of implementing a complex task-based model switching mechanism within the agent, we will focus on a configuration-based approach. The user will be able to define a collection of SLMs in the configuration and select one to be used for a given run.

This approach has several advantages:
-   **Simplicity**: It requires minimal changes to the core agent logic.
-   **Flexibility**: Users can easily define and switch between different SLMs without touching the code.
-   **Reusability**: It leverages the power of `litellm` and the existing `LiteLLMModel` class.

The following phases will detail how to implement this strategy.

## Phase 1: Configuration System Updates

The first step is to update the configuration system to support a registry of SLMs.

### 1.1 Add SLM Registry to Model Config

We will add a new field to the main model configuration to hold a dictionary of SLM provider configurations.

**File: `sweagent/agent/models.py`**

```python
// ... existing imports ...

class GenericAPIModelConfig(PydanticBaseModel):
    # ... existing fields ...
    model_config = ConfigDict(extra="forbid")

class ModelConfig(PydanticBaseModel):
    """The model configuration. This is the top-level model configuration that is part of the agent config."""

    # The sub_configs field is a union of all possible model configs
    sub_config: Annotated[
        Union[
            GenericAPIModelConfig,
            HumanModelConfig,
            HumanThoughtModelConfig,
            ReplayModelConfig,
            InstantEmptySubmitModelConfig,
        ],
        Field(discriminator="name"),
    ]
    slm_providers: dict[str, GenericAPIModelConfig] = Field(
        default_factory=dict,
        description="A dictionary of available SLM providers.",
    )

// ... rest of the file ...
```
*Note: The actual top-level model config is defined in `sweagent/utils/config.py` as `AgentConfig.model`. For clarity, the above change is presented in the context of `models.py`, but will be implemented in the appropriate location.*

### 1.2 Update Default Configuration File

Next, we'll add a section for SLMs to the default YAML configuration.

**File: `config/default.yaml`**

```yaml
# ... existing agent configuration ...
agent:
  model:
    slm_providers:
      my-local-slm:
        name: "local-model"
        api_base: "http://localhost:11434/v1"
        api_key: "dummy"
        max_input_tokens: 4096
        max_output_tokens: 1024
        temperature: 0.1
        top_p: 0.95
```

## Phase 2: CLI and Execution Flow Integration

Now we'll modify the main execution script to allow selecting an SLM via a command-line argument.

### 2.1 Add CLI Argument

We'll add a `--slm` argument to `run_single.py`.

**File: `sweagent/run/run_single.py`**

```python
import simple_parsing
# ... other imports

@dataclass
class RunArguments:
    # ... existing arguments
    slm: str = simple_parsing.field(
        default=None,
        description="The name of the SLM provider to use from the slm_providers config.",
    )

# In the main function:
def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(RunArguments, dest="run_args")
    parser.add_arguments(AgentConfig, dest="agent_config")
    parser.add_arguments(EnvironmentConfig, dest="env_config")
    args = parser.parse_args()

    run_args: RunArguments = args.run_args
    agent_config: AgentConfig = args.agent_config
    env_config: EnvironmentConfig = args.env_config

    # SLM selection logic
    if run_args.slm:
        if run_args.slm not in agent_config.model.slm_providers:
            raise ValueError(f"SLM provider '{run_args.slm}' not found in configuration.")
        
        # Override the main model config with the SLM config
        slm_config = agent_config.model.slm_providers[run_args.slm]
        agent_config.model.sub_config = slm_config

    # ... rest of the main function (agent and env instantiation) ...
```
This logic will replace the default model configuration with the selected SLM's configuration before the agent is created, seamlessly integrating the SLM into the existing workflow.

## Phase 3: Documentation

To ensure users can leverage this new functionality, we need to update the documentation.

### 3.1 Update Model Configuration Docs

**File: `docs/reference/model_config.md`**

We will add a section explaining how to configure SLMs.

```markdown
## Configuring Small Language Models (SLMs)

SWE-agent supports the use of custom Small Language Models (SLMs) that can be accessed via a `litellm`-compatible API endpoint. You can define a dictionary of SLMs in your configuration and select one at runtime.

### Defining SLM Providers

In your YAML configuration file, you can add an `slm_providers` dictionary to your model configuration:

```yaml
agent:
  model:
    slm_providers:
      my-local-slm:
        name: "local-model"
        api_base: "http://localhost:11434/v1" # e.g., for Ollama
        api_key: "dummy"
        # other GenericAPIModelConfig parameters...
      another-slm:
        name: "custom-slm"
        api_base: "https://api.custom-slm.com/v1"
        api_key: "$CUSTOM_SLM_API_KEY"
```

### Selecting an SLM

You can select an SLM for a run using the `--slm` command-line argument:

```bash
python -m sweagent.run.run_single --slm my-local-slm --problem_statement ...
```
This will use the `my-local-slm` configuration for the agent's model.
```

## Phase 4: Testing

We will add tests to ensure the SLM selection logic works correctly.

### 4.1 Create SLM Integration Test

**File: `tests/test_run_single.py`**

We can add a test case to verify that the `--slm` flag correctly overrides the model configuration.

```python
def test_slm_selection():
    # 1. Create a dummy config file with an SLM provider.
    # 2. Run run_single.py with the --slm flag.
    # 3. Mock the Agent class constructor.
    # 4. Assert that the Agent is initialized with the correct model configuration from the SLM provider.
    pass
```

## Phase 5: Future Enhancements (Out of Scope for Initial Implementation)

While the configuration-based approach is robust and flexible, more advanced integrations are possible. These are noted here as potential future work.

### 5.1 Dynamic Model Switching

- **Concept**: Allow the agent to switch between different models (e.g., a powerful general model and a specialized SLM) during a single run based on the current task.
- **Complexity**: This would be a significant architectural change. It would require:
    - A mechanism within the `DefaultAgent` to classify the current sub-task (e.g., "analyzing code", "writing a patch", "running tests").
    - A way to manage multiple model instances within the agent.
    - Logic to route queries to the appropriate model.
- **Feasibility**: High complexity. This would deviate from the current simple, single-model design of the agent.

### 5.2 SLM Health Checks

- **Concept**: A utility to ping the configured SLM endpoints to ensure they are available before starting a run.
- **Implementation**: This could be a separate script or a pre-run check in `run_single.py` that makes a simple API call (e.g., listing models) to the SLM's `api_base`.
- **Feasibility**: Low complexity. A useful utility for improving user experience.

This plan provides a clear, non-intrusive path to integrating SLMs into `swe-agent`, empowering users to leverage specialized models for their software engineering tasks.