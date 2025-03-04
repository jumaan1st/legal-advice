const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const mammoth = require('mammoth');

// Dynamic import for @xenova/transformers to resolve ESM issues
async function loadTransformers() {
  const { pipeline } = await import('@xenova/transformers');
  return pipeline;
}

const app = express();
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(bodyParser.json());
app.use(express.static('public'));

let embeddingModel;
let generationModel;
let documents = [];

// Cosine similarity function
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
}

// Function to split text into overlapping chunks
function splitTextIntoChunks(text, chunkSize = 500, overlap = 100) {
  const words = text.split(' ');
  let chunks = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    chunks.push(words.slice(i, i + chunkSize).join(' '));
  }
  return chunks;
}

// Load models and process documents
async function loadModelsAndData() {
  try {
    const pipeline = await loadTransformers();
    console.log('Loading embedding model...');
    embeddingModel = await pipeline('feature-extraction', 'Xenova/all-mpnet-base-v2');
    console.log('Embedding model loaded.');
    console.log('Loading generation model...');
    generationModel = await pipeline('text-generation', 'Xenova/distilgpt2');
    console.log('Generation model loaded.');

    const docsDir = path.join(__dirname, 'legal-docs');
    if (!fs.existsSync(docsDir)) {
      console.warn('legal-docs directory not found, creating it...');
      fs.mkdirSync(docsDir);
    }
    const files = fs.readdirSync(docsDir).filter(file => file.endsWith('.docx'));
    console.log(`Found ${files.length} .docx files`);
    
    for (const file of files) {
      const filePath = path.join(docsDir, file);
      console.log(`Processing ${file}...`);
      const result = await mammoth.extractRawText({ path: filePath });
      const text = result.value;
      const chunks = splitTextIntoChunks(text);
      
      for (const chunk of chunks) {
        const embedding = await embeddingModel(chunk, { pooling: 'mean', normalize: true });
        documents.push({ text: chunk, embedding: embedding.data });
      }
      console.log(`Processed ${file} into ${chunks.length} chunks.`);
    }
    console.log(`Loaded ${documents.length} document chunks.`);
  } catch (error) {
    console.error('Error loading models or documents:', error);
  }
}

app.get('/', (req, res) => {
  res.render('index');
});

app.post('/ask', async (req, res) => {
  try {
    const query = req.body.query;
    if (!embeddingModel || !generationModel) throw new Error('Models not loaded');

    const queryEmbedding = await embeddingModel(query, { pooling: 'mean', normalize: true });
    const queryVec = queryEmbedding.data;

    const similarities = documents.map(doc => ({
      text: doc.text,
      similarity: cosineSimilarity(queryVec, doc.embedding)
    }));
    
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topK = similarities.slice(0, 5);
    
    const prompt = `You are an AI legal assistant. Based on the most relevant legal texts, answer this: "${query}"\n\nRelevant Legal Texts:\n${topK.map(item => `- ${item.text}`).join('\n')}\n\nAnswer:`;
    
    const generated = await generationModel(prompt, { max_length: 150 });
    res.json({ response: generated[0].generated_text });
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

loadModelsAndData().then(() => {
  app.listen(3000, () => console.log('Server running on http://localhost:3000'));
}).catch(err => console.error('Failed to start server:', err));
