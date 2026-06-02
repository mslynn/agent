// import { ChatOpenAI } from "@langchain/openai";

// const model = new ChatOpenAI({
//   modelName: "deepseek-ai/DeepSeek-V4-Pro",
//   apiKey: "sk-pocfmfsknpgomaivhwhvngheoqiolqgtmetwslntfbaspwyr",
//   configuration: {
//     baseURL: "https://api.siliconflow.cn/v1",
//   },
// });

// const response = await model.invoke("介绍下自己");
// console.log(response.content);


import dotenv from'dotenv';
import { ChatOpenAI } from'@langchain/openai';

dotenv.config();

const model = new ChatOpenAI({ 
    modelName: process.env.MODEL_NAME || "qwen-coder-turbo",
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const response = await model.invoke("介绍下自己");
console.log(response.content);
