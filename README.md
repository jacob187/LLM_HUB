# Large Language Model Hub

## Description

My goal for this project is to create a universal interface for interacting with LLMs by leveraging LangChain. Ultimately, I want to create a modular structure where I can build APIs to integrate LLMs into other applications.

The Streamlit application allows you to interact with different LLMs. Currently, the project supports Anthropic and OpenAI models.

## Installation

I am using poetry to manage python dependencies. Ensure that you have Python v3.11.9 or newer on your machine.

Install the poetry package manager:

```bash
pip install poetry
```

Then install the required packages:

```bash
poetry install
```

Add your own API keys and python path to a `.env` file. See `.env.example` for reference.

Add the path to this python project in the `PYTHONPATH` variable in `.env`

To activate the virtual environment, run the following command:

```bash
poetry shell
```

If you wish to run the Streamlit application, run the following command:

```bash
streamlit run llm_app/frontend/app.py
```

## Current Structure

### Models

I have used inheritance to create a LLM model in `llm_app/backend/llms`. I'm using the `BaseLLM` to create new LLM classes as desired. I have created a chat manger at `llm_app/backend/chat`, as well as a memory and database mangers, both of which are still in development.

## License

This project is licensed under the MIT License.

## Contributing

Please feel free to submit a Pull Request! I welcome any contributions.
