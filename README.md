# Open AI Overview
This repository is meant to remove the fog and provide clearance on key concepts of OpenAI and AI topics in general.
It contains information about the history of OpenAI. The modules and models, and use cases.

It is based on my understanding, knowledge, and experience.
The information was summarized after a lot of reading, yet I tried to make it as informative as possible.

I hope you will find it informative.


## Key acronyms and subjects:
**Artificial Intelligent (AI):**
Artificial intelligence (AI) involves creating computer systems that can perform tasks requiring human intelligence. It encompasses techniques like machine learning, natural language processing, and computer vision. AI aims to simulate human intelligence for tasks such as understanding language, recognizing images, and problem-solving. AI applications are found in healthcare, finance, transportation, and more. While true human-level intelligence is a long-term goal, AI technologies are already making a significant impact on society, driving innovation across industries.

**LLM:**
Large Language Models (LLMs) are advanced AI models trained on vast amounts of text data. They generate human-like text and perform language-related tasks. 
LLMs, like GPT-3 and GPT-4, use deep learning and learn to predict the next word based on context. They excel in tasks like text generation, translation, summarization, and question answering. LLMs find applications in virtual assistants, content generation, language translation, sentiment analysis, and chatbots. However, they are not flawless and may produce incorrect or biased information, necessitating human oversight.

**Neural Network:**
Neural networks are a type of artificial intelligence (computational models) that use a system of algorithms to process data and recognize patterns. They are modeled after the human brain and consist of interconnected nodes called neurons arranged in layers. 

They are used to solve complex problems such as image recognition, natural language processing, and robotics. Neural networks have been used in a variety of applications, including medical diagnosis, autonomous vehicles, and financial forecasting.



These networks learn patterns and relationships in data by adjusting weights on connections between neurons. Neural networks can recognize complex patterns, process input data, and produce outputs based on learned information. They are used in various fields like image recognition, speech processing, and recommendation systems. Neural networks excel at tasks such as image classification, language translation, and decision-making. Overall, they are powerful tools for solving complex problems with large amounts of data.


![Alt text][id0]

[id0]: https://community-cdn.rstudio.com/uploads/default/original/2X/b/be26d200b29312ebef7e1d8ec361f25e5305852c.jpeg


**GPT**:
GPT (Generative Pre-trained Transformer) model is an advanced language generation model developed by OpenAI. It uses self-attention mechanisms to understand the context of text and has been trained on vast amounts of data from the internet. 
    &nbsp;
    GPT models can generate coherent and contextually relevant text by predicting the next word or phrase based on the preceding context. They excel in tasks such as story generation, question answering, translation, summariztion and more. GPT models have numerous applications and are valued for their language understanding and generation capabilities, making them useful for developers and users alike.

![Alt text][id1]

[id1]: https://uploads-ssl.webflow.com/627a5f477d5ec9079c88f0e2/63be984917f71c97903b3e90_ChatGPT%20training%20diagram.png "ChatGPT Traiining Diagram"




**Hugging Face:**

[Hugging Face](https://huggingface.co/) is an OpenSource company focusing on NLP (Natural Language Processing) and developing various tools and libraries for working with NLP.
One of their most well-known contributions is the development of the *Transformers* library, which has become widely adopted in the NLP community.

In addition to the Transformers library, Hugging Face offers the Hugging Face Hub, a platform where researchers and developers can share and discover pre-trained models, datasets, and other resources. The hub provides a centralized repository for NLP models and fosters collaboration within the NLP community.

The [Transformers library](https://huggingface.co/docs/transformers/index) provides a high-level API and pre-trained models for various NLP tasks, including text classification, named entity recognition, question answering, language translation, and more. It is built on top of PyTorch and TensorFlow, allowing users to easily leverage state-of-the-art models such as BERT, GPT, RoBERTa, and others.

The Hugging Face [Model Hub](https://huggingface.co/docs/hub/models-the-hub) is a central repository where users can discover, download, and fine-tune pre-trained models for their specific NLP tasks. The model hub hosts a vast collection of models.  It serves as a one-stop shop for accessing and utilizing cutting-edge NLP models, reducing the barrier to entry for developers.

Hugging Face provides high-level [NLP pipelines](https://huggingface.co/docs/transformers/main/main_classes/pipelines) that encapsulate complex processes like text classification, named entity recognition, and sentiment analysis. These pipelines abstract away the technical details, allowing users to perform NLP tasks with just a few lines of code. This simplification has made it easier for developers to integrate NLP capabilities into their applications.

You can learn more and see all their repositories on their [Github](https://github.com/huggingface) page.

---
&nbsp;

![Alt text][id00]

[id00]: https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/2560px-OpenAI_Logo.svg.png

&nbsp;



## **OpenAI**:

[OpenAI is an artificial intelligence research organization](https://openai.com/) that aims to ensure that artificial general intelligence (AGI) benefits all of humanity. 

It conducts research, develops AI models and technologies, and promotes the principles of transparency, safety, and ethical use of AI. OpenAI has created advanced language models, such as GPT-3, which can generate human-like text and have a wide range of applications. 

The organization also provides access to its models and resources through [APIs](https://platform.openai.com/docs/api-reference/models/retrieve), enabling developers to leverage its AI capabilities. OpenAI's mission is to advance AI technology while prioritizing its responsible and beneficial deployment for the betterment of society.


### OpenAI Models (popular):
1. **[GPT-3](#GPT) (Generative Pre-trained Transformer 3):**
    GPT (Generative Pre-trained Transformer) model is an advanced language generation model developed by OpenAI. It uses self-attention mechanisms to understand the context of text and has been trained on vast amounts of data from the internet. 
    &nbsp;
    GPT models can generate coherent and contextually relevant text by predicting the next word or phrase based on the preceding context. They excel in tasks such as story generation, question answering, translation, summariztion and more. GPT models have numerous applications and are valued for their language understanding and generation capabilities, making them useful for developers and users alike.
&nbsp;

2. **DALL-E:** 
    DALL·E is a 12-billion parameter version of GPT-3 trained to generate images from text descriptions using a dataset of text–image pairs (generative AI model).

    **DALL·E 2** is the 2nd generation model that can create more realistic (4x) images and art from a description in natural language.
&nbsp;

3. **Codex:**
[Codex](https://platform.openai.com/docs/guides/code) is the AI model that writes code based on natural language prompts. It is the model used today in [Github Copilot](https://github.com/features/copilot). It supports many languages, including JavaScript, Python, Go, Perl, Ruby, PHP, Swift, Shell, and more.
&nbsp;


4. **CLIP (Contrastive Language-Image Pre-training):** 
    [CLIP](https://openai.com/research/clip)  is a powerful deep-learning model that can understand and connect images and text. CLIP has been trained on a vast amount of text and image data to learn their relationship.
&nbsp;
    CLIP is useful for several reasons; It has the ability to use "zero-shot learning" to perform tasks without any specific training for that particular task. it also supports "few-shot training", and can be applied across different domains (Cross-domain Applications), such as natural images, artwork, product images, and more.
&nbsp;

### GPT versions (common):
**GPT3**  was the 3rd and the version that brought OpenAI and ChatGPT to people's conscience. GPT 3 was trained with 45TB of text data, including sources like books, Wikipedia, filtered Common Crawls data, Webtext, and more.

*GPT 3* helped to train and develop the [DALL-E](https://openai.com/research/dall-e) (creates images from text), [Whisper](https://openai.com/research/whisper) (connects text and images), [CLIP](https://openai.com/research/clip) (multi-lingual voice-to-text), and [ChatGPT](https://openai.com/blog/chatgpt) (the most productive co-pilot :robot:) Models.


**GPT4 (released in 14.3.23):** 
GPT 4 is around ten times more advanced than GPT 3.5. it has a maximum token limit of 32,000 compared to GPT3.5, which had 4,000 tokens.


**History:**

* GPT1 - Trained with 4.5GB of text from 7000 unpublished (fiction) [BooksCorpus](https://paperswithcode.com/dataset/bookcorpus) books dataset that was primarily meant to understand language.
&nbsp;
* GP2 - Was trained with 40GB of text data and had 1.5 billion parameters.
&nbsp;
* GPT3 - Trained with 45TB of text data including sources like books, Wikipedia, [Common Crawls](https://commoncrawl.org/) , Webtext, and more. 175 billion parameters
&nbsp;
* GPT4 - Have 100 trillion parameters. Training data information wasn't disclosed formally. However,Andrew Feldman, the head of Cerebras, which is collaborating with OpenAI on the GPT model training, is quoted as saying that is the amount for the 4th version model.

&nbsp;

**Training and parameters comparison:**
| GPT Version | Trained by | Datasets / Data sources used | Parameters count | Release Date |
| :-----------: | :-----------: |:-----------: |:-----------:|:-----------: |
| GPT-1 | 4.5GB of data |BookCorpus: 4.5 GB of text, from 7,000 unpublished books| 117 Million Parameters|  2018|
| GPT-2 | 40 GB of data |WebText: 40 GB of text, 8 million documents, from 45 million webpages upvoted on Reddit.| 1.5 Billion Parameters| 14.02.2019 |
| GPT-3 | 570 GB | 570GB plaintext, 0.4 trillion tokens. Mostly [CommonCrawl](https://commoncrawl.org/), WebText, English Wikipedia, and two books corpora (Books1 and Books2).| 175 Billion Parameters| 2020 |
| GPT-3.5 | 45 TB of data | Finetuned version of GPT3| 175 Billion Parameters| 15.03.2022 |
| GPT-4 | undisclosed |trained data and parameters info are officialy undiscloosed yet but there are rumors that indicates those numbers| 100 Trillion Parameters| 14.03.2023 |




### ChatGPT:

ChatGPT is an application developed from the GPT-3.5 model, utilizing GPT as its language model. It offers a user-friendly, chat-like interface that enables easy interaction, resembling a conversation with a human.

Though ChatGPT possesses immense power and extensive knowledge, it may occasionally provide hallucinative answers. However, despite this drawback, ChatGPT remains the most advanced "Assistant" available today.

Conversations with ChatGPT begin with a user-provided prompt; from there, the sky(net) is the limit. 

[//]: <> (https://thumbs.gfycat.com/DefensiveCarefreeKawala-max-1mb.gif)

![color picker](https://i.pinimg.com/originals/3b/4b/43/3b4b43255b54024c5626e3d34dbeee1e.gif 'Skynet Theory')



### GPT3 [Models](https://platform.openai.com/docs/models/overview):
GPT-3 models can understand and generate natural language. These models were superseded by the more powerful GPT-3.5 generation models. However, the original GPT-3 base models (`davinci`, `curie`, `ada`, and `babbage`) are currently the only models that are available to fine-tune.
- **Ada**: 	Capable of simple tasks, usually the fastest model in the GPT-3 series, and lowest cost.
- **Babbage**: Capable of straightforward tasks, very fast, and lower cost.
- **Currie**: Very capable, but faster and lower cost than Davinci.
- **Davinci**: Most capable GPT-3 model. It can do any task the other models can do, often with higher quality. It has a great summarization engine and creates creative content.
        
The most common (but not only) interaction option with GPT is by providing a prompt.

### Prompt: 
The **prompt** is the basic interaction with OpenAI. it is how we ask the engine to answer, sort, list " act as" and more...

For example:
```
prompt -  
how to make a pizza?

prompt - 
what is the name of IronMan? 

prompt - 
who is buzz lightyear?

prompt -  
explain this code in human readable language:
code content 

prompt -  
summarize this email in 3 rows:
email content
```

### Completions:
OpenAI Completions is simply what we knew as [GPT](#GPT-3) (Generative Pre-trained Transformer). 


### Prompt Engineering
Prompt engineering is the known term of "crafting" effective inputs or prompts to get the desired output response from a language model like OpenAI (completions).
The completion will provide a result based on the "instruction" or behavior you provide in your prompt.

Prompt engineering is very important since LLM's rely heavily on the input they receive to generate text. 
By providing a well-crafted prompt, consumers can guide the model's output toward the desired outcome, improve the quality of responses, and control the model's behavior.

The main elements of a prompt are:
 <span style='color: blue;'>Instructions</span>, <span style='color: green;'>Context</span>, <span style='color: purple;'>Input data</span>, <span style='color: orange;'>Output indicator</span>


Example Prompt:

<span style='color: blue;'>Translate this text to Spanish: </span>
<span style='color: green;'>text: </span><span style='color: purple;'>Hello world! </span>

<span style='color: orange;'>Response: Hola mundo! </span>


While using prompts, there are many types of behaviors you want / can include in your prompt, like:


    * Provide instruction: summarize, write, translate, order,and explain..

    * Specify: "write it in 5 rows". "write it in a formal language", "provide many details", etc...

    * Generate sample data - generate sample / synthetic data for building a dummy database or text files or information that can be useful for Labs.

    * Impersonate - **"act as"** [role/person/type].

    * Story telling - answer in a "storytelling" way.

    * Code generation: "write a python code of..."

    * Placeholders - use place holder to write a story, generate data, etc.. 
    Usage example: the "{text input here}" is a placeholder for actual text/context 

    * "Few shots"  - provide input in your instructions to "train" the model to understand an acronym / a word.

    * CoT(Chain of Thought) - Provide guidelines to solve a quiz or an enigma. and combine instructions (algorithm) to get the desired result.


You can find more recommendations on the OpenAI [Best Practices thread](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)


I like to use my prompts using the "When, What, How (optional: who)" guideline. It can help you to get more accurate completions. for example. 
    `prompt: tell me  how was your morning? "act as BA baracus"`


**Effective prompt-engineering considerations:**
1. **Clear instructions** - provide explicit instructions or guidelines to the model about the desired task or behavior. It should specify the format, context, or type of response expected.
2. **Conditioning** - By conditioning the prompt on relevant context or information, you can guide the model to generate responses that are specific to a particular domain or topic.
3. **Bias mitigation** - Prompt engineering can help address biases that may be present in the language model. By carefully designing prompts & using techniques like counterfactuals or neutralizing language, users can reduce bias in the model's responses.
4. **Systematic testing** - Experimenting with different prompts and evaluating the model's responses can help refine and improve prompt engineering. Iterative testing and analysis can lead to a better understanding of the model's capabilities and limitations.


**To sumarize:**
Prompt engineering is how we ask the LLM's to perform a specific task.
 It is an ongoing process that requires experimentation, refinement, and adaptation based on the specific task or application.
&nbsp;
&nbsp;

### Fine-tunning:
[Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) refers to the process of further training a pre-trained language model, such as GPT, on a specific dataset or task to improve its performance and adapt it to a specific domain or application.

The fine-tuning process involves taking a pre-trained model, which has been trained on a large corpus of text from the internet, and then continuing the training on a smaller, domain-specific dataset. This allows the model to learn the specific patterns, language nuances, and context relevant to the target domain.

During fine-tuning, the model is exposed to the new dataset, and the parameters of the model are updated through gradient-based optimization methods like backpropagation. The objective is to adjust the model's parameters to better align with the target task or dataset, optimizing its performance on that specific task.

### Embedding:
[Embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) are mainly used for search, clustering, recommendations, anomaly detection, and classification. 

An embedding is a `vector` (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

The Embedding response can look like this:

    {
    "data": [
        {
        "embedding": [
            -0.006929283495992422,
            -0.005336422007530928,
            ...
            -4.547132266452536e-05,
            -0.024047505110502243
        ],
        "index": 0,
        "object": "embedding"
        }
    ],
    "model": "text-embedding-ada-002",
    "object": "list",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
    }


----

### Tokens and Tokens limitations
Tokens in OpenAI are units of text used during language processing. 
While "prompting", before to the API process of the prompts, the input is broken down into *tokens*. 

These tokens are calculated based on where words start or end.
Tokens can include trailing spaces and even sub-words. 

Here are a few facts about tokens:
- They can be as short as a single character or as long as a word or phrase
- Tokenization is the process of dividing the text into tokens
- Each token is assigned a numerical value for machine processing
- Longer words or phrases may be split into multiple tokens
- Tokenization affects the overall size and complexity of language models.
- The number of tokens used can impact the cost and performance of using OpenAI models

Here are some helpful rules of thumb for understanding tokens in terms of lengths:

- 1 token ~= 4 chars in English
- 100 tokens ~= 75 word
- 1 token ~= ¾ words

or
- 1-2 sentences ~= 30 tokens
- 1 paragraph ~= 100 tokens
- 1,500 words ~= 2048 tokens


You can calculate the expected token counts based on your prompt on this [Tokenizer](https://platform.openai.com/tokenizer), which is a **Tokens calculator**.

&nbsp;

---

![Alt text][id4]

[id4]: https://blogs.microsoft.com/wp-content/uploads/prod/2023/01/PNG-openai-microsoft_960x540.png "Microsoft + OpenAI"

### OpenAI and Azure:
OpenAI GPT models were trained on Microsoft [Azure](https://azure.microsoft.co) supercomputing. OpenAI runs on an AI-optimized infrastructure to provide the best performance to suit OpenAI models.

Microsoft is a big contributor and has big investments in OpenAI.

Microsoft also provides its own native, 1st party OpenAI service (AOAI== Azure OpenAI) that brings all the features and capabilities of OpenAI with extended capabilities for enterprises. 

It supports the following models:
* Generative AI - Text Models:
    * GPT-3.5
    * ChatGPT (Preview)
    * GPT-4 (Preview)
* Generative AI - Image Models:
    * DALL-E 2 (Preview)


**Using the Azure OpenAI service provides you with additional benefits like:**

* Deploy in your Azure subscription.
* Integration with other Azure services.
* More control over the configuration and performance.
* etc...



You can read more about Azure OpenAI on my other [repository](https://github.com/benarch/Azure/blob/main/Azure_OpenAI/Azure_OpenAI.md).

**References:**
[Azure OpenAI Produt page](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)
[Azure OpenAI Documentation page](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview)
[Azure OpenAI Pricing page](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)


![Alt text][id2]

[id2]: https://octodex.github.com/images/droidtocat.png "Keep Prompting"


**Note:
Posts and blogs are my own. it is my personal interpertation and doesn't conclude or provide any recomendation or formal suggestion.**

