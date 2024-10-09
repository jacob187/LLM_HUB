# Large Language Model Hub

## Description

My goal for this project is to create a universal interface for interacting with LLMs by leveraging langchain. Ultimatley, I want to create a modular structure where I can build APIs to integrate LLMs into other applications.

## Installation

I am using poetry to manage python dependencies. To install the required packages, run the following command:

```bash
poetry install
```

Add your own API keys to a `.env` file.

To activate the virtual environment, run the following command:

```bash
poetry shell
```

If you want to run the streamlit app, run the following command:

```bash
streamlit run llm_app/frontend/app.py
```

## Current Structure

### Models

I have used inheritance to create a LLM model in `llm_app/backend/llms`. I'm using the `BaseLLM` to create new LLM classes as desired.

## License

This project is licensed under the MIT License.

## Contributing

Please feel free to submit a Pull Request! I welcome any contributions.




