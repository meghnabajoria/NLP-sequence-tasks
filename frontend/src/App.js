import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [translationInput, setTranslationInput] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  
  const [summarizationInput, setSummarizationInput] = useState('');
  const [summarizedText, setSummarizedText] = useState('');

  const [qaInput, setQaInput] = useState({ question: '', context: '' });
  const [qaOutput, setQaOutput] = useState('');

  const handleTranslate = async () => {
    const response = await axios.post('http://localhost:5000/translate', { text: translationInput });
    setTranslatedText(response.data[0].translation_text);
  };

  const handleSummarize = async () => {
    const response = await axios.post('http://localhost:5000/summarize', { text: summarizationInput });
    setSummarizedText(response.data[0].summary_text);
  };

  const handleQa = async () => {
    const response = await axios.post('http://localhost:5000/qa', qaInput);
    setQaOutput(response.data.answer);
  };

  return (
    <div className="App">
      <h1>NLP Application</h1>
      
      <div className="task">
        <h2>Translation</h2>
        <textarea 
          value={translationInput} 
          onChange={(e) => setTranslationInput(e.target.value)} 
          placeholder="Enter text to translate"
        />
        <button onClick={handleTranslate}>Translate</button>
        <p>Translated Text: {translatedText}</p>
      </div>
      
      <div className="task">
        <h2>Summarization</h2>
        <textarea 
          value={summarizationInput} 
          onChange={(e) => setSummarizationInput(e.target.value)} 
          placeholder="Enter text to summarize"
        />
        <button onClick={handleSummarize}>Summarize</button>
        <p>Summarized Text: {summarizedText}</p>
      </div>

      <div className="task">
        <h2>Question Answering</h2>
        <textarea 
          placeholder="Question" 
          value={qaInput.question} 
          onChange={(e) => setQaInput({ ...qaInput, question: e.target.value })}
        />
        <textarea 
          placeholder="Context" 
          value={qaInput.context} 
          onChange={(e) => setQaInput({ ...qaInput, context: e.target.value })}
        />
        <button onClick={handleQa}>Get Answer</button>
        <p>Answer: {qaOutput}</p>
      </div>
    </div>
  );
}

export default App;
