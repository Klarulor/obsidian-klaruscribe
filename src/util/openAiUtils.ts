/**
 * This was heavily inspired by
 * https://github.com/drewmcdonald/obsidian-magic-mic
 * Thank you for traversing this in such a clean way
 */
import OpenAI from 'openai';
import audioDataToChunkedFiles from './audioDataToChunkedFiles';
import type { FileLike } from 'openai/uploads';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { SystemMessage } from '@langchain/core/messages';
import { GoogleGenAI } from '@google/genai';

import { Notice } from 'obsidian';
import type { ScribeOptions } from 'src';
import { LanguageOptions } from './consts';
import { convertToSafeJsonKey } from './textUtil';
import ScribePlugin from 'src';

export enum LLM_MODELS {
  'gpt-4.1' = 'gpt-4.1',
  'gpt-4.1-mini' = 'gpt-4.1-mini',
  'gpt-4o' = 'gpt-4o',
  'gpt-4o-mini' = 'gpt-4o-mini',
  'gpt-4-turbo' = 'gpt-4-turbo',
  'gemini-2.5-flash' = 'gemini-2.5-flash',
}

const MAX_CHUNK_SIZE = 25 * 1024 * 1024;

export async function chunkAndTranscribeWithGemini(
  geminiApiKey: string,
  audioBuffer: ArrayBuffer,
  { audioFileLanguage }: Pick<ScribeOptions, 'audioFileLanguage'>,
  customBaseUrl?: string,
  customModel?: string,
): Promise<string> {
  if (!geminiApiKey) {
    throw new Error('Gemini API key is required to transcribe audio.');
  }

  const geminiClient = new GoogleGenAI({ apiKey: geminiApiKey });
  if (customBaseUrl) {
    new Notice(
      'Scribe: custom Gemini base URLs are not supported; using default endpoint.',
    );
  }

  const modelName = customModel || LLM_MODELS['gemini-2.5-flash'];
  const audioFiles = await audioDataToChunkedFiles(audioBuffer, MAX_CHUNK_SIZE);
  new Notice(`Scribe: 🎧 Split transcript into ${audioFiles.length} files`);

  const transcriptChunks: string[] = [];
  for (const [chunkIndex, file] of audioFiles.entries()) {
    const chunkTranscript = await transcribeChunkWithGemini({
      file,
      client: geminiClient,
      modelName,
      chunkIndex,
      totalChunks: audioFiles.length,
      audioFileLanguage,
    });
    transcriptChunks.push(chunkTranscript);
  }

  return transcriptChunks.join(' ').trim();
}

export async function chunkAndTranscribeWithOpenAi(
  openAiKey: string,
  audioBuffer: ArrayBuffer,
  { audioFileLanguage }: Pick<ScribeOptions, 'audioFileLanguage'>,
  customBaseUrl?: string,
  customModel?: string,
) {
  const openAiClient = new OpenAI({
    apiKey: openAiKey,
    dangerouslyAllowBrowser: true,
    ...(customBaseUrl && { baseURL: customBaseUrl }),
  });
  const audioFiles = await audioDataToChunkedFiles(audioBuffer, MAX_CHUNK_SIZE);
  new Notice(`Scribe: 🎧 Split transcript into ${audioFiles.length} files`);

  const transcript = await transcribeAudio(openAiClient, {
    audioFiles,
    audioFileLanguage,
    customModel,
  });

  return transcript;
}

/**
 * Transcribe an audio file with OpenAI's Whisper model
 *
 * Handles splitting the file into chunks, processing each chunk, and
 * concatenating the results.
 */

interface TranscriptionOptions {
  audioFiles: FileLike[];
  onChunkStart?: (i: number, totalChunks: number) => void;
  audioFileLanguage?: LanguageOptions;
  customModel?: string;
}

async function transcribeAudio(
  client: OpenAI,
  { audioFiles, onChunkStart, audioFileLanguage, customModel }: TranscriptionOptions,
): Promise<string> {
  let transcript = '';
  for (const [i, file] of audioFiles.entries()) {
    if (onChunkStart) {
      onChunkStart(i, audioFiles.length);
    }

    const useAudioFileLanguageSetting =
      audioFileLanguage && audioFileLanguage !== LanguageOptions.auto;

    const modelToUse = customModel || 'whisper-1';
    const baseOptions = {
      model: modelToUse,
      file,
    };
    const whisperOptions = useAudioFileLanguageSetting
      ? { ...baseOptions, language: audioFileLanguage }
      : baseOptions;

    const res = await client.audio.transcriptions.create(whisperOptions);
    const sep = i === 0 ? '' : ' ';
    transcript += sep + res.text.trim();
  }
  return transcript;
}

interface GeminiChunkTranscriptionOptions {
  file: FileLike;
  client: GoogleGenAI;
  modelName: string;
  chunkIndex: number;
  totalChunks: number;
  audioFileLanguage?: LanguageOptions;
}

async function transcribeChunkWithGemini({
  file,
  client,
  modelName,
  chunkIndex,
  totalChunks,
  audioFileLanguage,
}: GeminiChunkTranscriptionOptions): Promise<string> {
  const inlineDataPart = await fileLikeToInlineData(file);
  const languageInstruction =
    audioFileLanguage && audioFileLanguage !== LanguageOptions.auto
      ? ` The audio is spoken in ${audioFileLanguage}.`
      : '';

  const result = await client.models.generateContent({
    model: modelName,
    contents: [
      {
        role: 'user',
        parts: [
          {
            text:
              `Transcribe audio chunk ${chunkIndex + 1} of ${totalChunks}.${languageInstruction} ` +
              'Return only the verbatim transcript without timestamps, speaker labels, or commentary.',
          },
          inlineDataPart,
        ],
      },
    ],
  });

  const text = result?.text?.trim();
  if (!text) {
    throw new Error('Gemini returned an empty transcription chunk.');
  }

  return text;
}

interface GeminiInlineDataPart {
  inlineData: {
    data: string;
    mimeType: string;
  };
}

async function fileLikeToInlineData(file: FileLike): Promise<GeminiInlineDataPart> {
  const blob = await fileLikeToBlob(file);
  const data = await blobToBase64(blob);
  const mimeType = blob.type && blob.type.length > 0 ? blob.type : inferMimeType(file);

  return {
    inlineData: {
      data,
      mimeType,
    },
  };
}

async function fileLikeToBlob(file: FileLike): Promise<Blob> {
  if (isBlobLike(file)) {
    return file as unknown as Blob;
  }

  const possibleData = (file as { data?: unknown }).data;
  if (possibleData) {
    if (isBlobLike(possibleData)) {
      return possibleData as Blob;
    }
    if (possibleData instanceof ArrayBuffer) {
      return new Blob([possibleData], { type: inferMimeType(file) });
    }
    if (ArrayBuffer.isView(possibleData)) {
      const view = possibleData as ArrayBufferView;
      const copy = new Uint8Array(view.byteLength);
      copy.set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
      return new Blob([copy.buffer], { type: inferMimeType(file) });
    }
  }

  throw new Error('Unable to convert audio chunk into Blob for Gemini transcription.');
}

async function blobToBase64(blob: Blob): Promise<string> {
  const arrayBuffer = await blob.arrayBuffer();
  return arrayBufferToBase64(arrayBuffer);
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';

  for (let i = 0; i < bytes.byteLength; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  if (typeof btoa === 'function') {
    return btoa(binary);
  }

  const nodeBufferCtor = (globalThis as {
    Buffer?: {
      from(data: ArrayBufferLike | ArrayLike<number>): {
        toString(encoding: 'base64'): string;
      };
    };
  }).Buffer;

  if (nodeBufferCtor) {
    return nodeBufferCtor.from(bytes).toString('base64');
  }

  throw new Error('Base64 encoding is not supported in the current environment.');
}

function inferMimeType(file: FileLike): string {
  const fileWithMeta = file as File & { type?: string };
  if (fileWithMeta.type && fileWithMeta.type.length > 0) {
    return fileWithMeta.type;
  }
  if ('name' in fileWithMeta && typeof fileWithMeta.name === 'string') {
    if (fileWithMeta.name.toLowerCase().endsWith('.mp3')) {
      return 'audio/mpeg';
    }
    if (fileWithMeta.name.toLowerCase().endsWith('.wav')) {
      return 'audio/wav';
    }
  }
  return 'audio/wav';
}

function isBlobLike(value: unknown): value is Blob {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as Blob).arrayBuffer === 'function'
  );
}

export async function summarizeTranscript(
  openAiKey: string,
  transcript: string,
  scribeOptions: ScribeOptions,
  llmModel: LLM_MODELS = LLM_MODELS['gpt-4o'],
  customBaseUrl?: string,
  customChatModel?: string,
) {
  const systemPrompt = buildSummarySystemPrompt(transcript);
  const modelToUse = customChatModel || llmModel;
  const model = new ChatOpenAI({
    model: modelToUse,
    apiKey: openAiKey,
    temperature: 0.5,
    ...(customBaseUrl && { configuration: { baseURL: customBaseUrl } }),
  });
  const messages = [new SystemMessage(systemPrompt)];

  if (scribeOptions.scribeOutputLanguage) {
    messages.push(
      new SystemMessage(
        `Please respond in ${scribeOptions.scribeOutputLanguage} language`,
      ),
    );
  }

  const structuredOutput = buildSummarySchema(scribeOptions);
  const structuredLlm = model.withStructuredOutput(structuredOutput);
  const result = (await structuredLlm.invoke(messages)) as Record<
    string,
    string
  > & { fileTitle: string };

  return await result;
}

export async function summarizeTranscriptWithGemini(
  geminiApiKey: string,
  transcript: string,
  scribeOptions: ScribeOptions,
  llmModel: LLM_MODELS = LLM_MODELS['gemini-2.5-flash'],
  customBaseUrl?: string,
) {
  if (!geminiApiKey) {
    throw new Error('Gemini API key is required to summarize the transcript.');
  }

  const geminiClient = new GoogleGenAI({ apiKey: geminiApiKey });
  if (customBaseUrl) {
    new Notice(
      'Scribe: custom Gemini base URLs are not supported; using default endpoint.',
    );
  }

  const fields = buildSummaryFields(scribeOptions);
  const schema = buildSummarySchema(scribeOptions);
  const systemPrompt = buildSummarySystemPrompt(transcript);
  const jsonInstructions = buildJsonInstructions(fields);
  const fullPrompt = `${systemPrompt}

${jsonInstructions}`;

  const response = await geminiClient.models.generateContent({
    model: llmModel,
    contents: [
      {
        role: 'user',
        parts: [
          {
            text: fullPrompt,
          },
        ],
      },
    ],
  });

  const textResponse = sanitizeGeminiResponse(response.text?.trim());
  if (!textResponse) {
    throw new Error('Gemini returned an empty summary response.');
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(textResponse);
  } catch (error) {
    throw new Error('Gemini summary response was not valid JSON.');
  }

  return schema.parse(parsed);
}

export async function llmFixMermaidChart(
  openAiKey: string,
  brokenMermaidChart: string,
  llmModel: LLM_MODELS = LLM_MODELS['gpt-4o'],
  customBaseUrl?: string,
  customChatModel?: string,
) {
  const systemPrompt = `
You are an expert in mermaid charts and Obsidian (the note taking app)
Below is a <broken-mermaid-chart> that isn't rendering correctly in Obsidian
There may be some new line characters, or tab characters, or special characters.  
Strip them out and only return a fully valid unicode Mermaid chart that will render properly in Obsidian
Remove any special characters in the nodes text that isn't valid.

<broken-mermaid-chart>
${brokenMermaidChart}
</broken-mermaid-chart>

Thank you
  `;
  const modelToUse = customChatModel || llmModel;
  const model = new ChatOpenAI({
    model: modelToUse,
    apiKey: openAiKey,
    temperature: 0.3,
    ...(customBaseUrl && { configuration: { baseURL: customBaseUrl } }),
  });
  const messages = [new SystemMessage(systemPrompt)];
  const structuredOutput = z.object({
    mermaidChart: z.string().describe('A fully valid unicode mermaid chart'),
  });

  const structuredLlm = model.withStructuredOutput(structuredOutput);
  const { mermaidChart } = await structuredLlm.invoke(messages);

  return { mermaidChart };
}

interface SummaryField {
  key: string;
  description: string;
  optional?: boolean;
}

function buildSummaryFields({ activeNoteTemplate }: ScribeOptions): SummaryField[] {
  const baseField: SummaryField = {
    key: 'fileTitle',
    description: `A suggested title for the Obsidian Note. Разрешенные символы a-Z и а-Я, и пробел. Используй язык ${ScribePlugin.globalConfig.noteFilenameLanguage}.`,
    optional: false,
  };

  const sectionFields = activeNoteTemplate.sections.map((section) => ({
    key: convertToSafeJsonKey(section.sectionHeader),
    description: section.sectionInstructions,
    optional: section.isSectionOptional,
  }));

  return [baseField, ...sectionFields];
}

function buildSummarySchema(options: ScribeOptions) {
  const fields = buildSummaryFields(options);
  const schemaShape: Record<string, z.ZodType<string | null | undefined>> = {};

  fields.forEach((field) => {
    schemaShape[field.key] = field.optional
      ? z.string().nullish().describe(field.description)
      : z.string().describe(field.description);
  });

  return z.object(schemaShape);
}

function buildSummarySystemPrompt(transcript: string) {
  return `
  You are "Scribe" an expert note-making AI for Obsidian you specialize in the Linking Your Thinking (LYK) strategy.  
  The following is the transcription generated from a recording of someone talking aloud or multiple people in a conversation. 
  There may be a lot of random things said given fluidity of conversation or thought process and the microphone's ability to pick up all audio.  

  The transcription may address you by calling you "Scribe" or saying "Hey Scribe" and asking you a question, they also may just allude to you by asking "you" to do something.
  Give them the answers to this question

  Give me notes in Markdown language on what was said, they should be
  - Easy to understand
  - Succinct
  - Clean
  - Logical
  - Insightful
  
  It will be nested under a h2 # tag, feel free to nest headers underneath it
  Rules:
  - Do not include escaped new line characters
  - Do not mention "the speaker" anywhere in your response.  
  - The notes should be written as if I were writing them. 
  - Пиши только на русском языке, за иссключением аббревиатур!

  The following is the transcribed audio:
  <transcript>
  ${transcript}
  </transcript>
  `;
}

function buildJsonInstructions(fields: SummaryField[]) {
  const fieldLines = fields
    .map((field) => {
      const optional = field.optional ? ' (optional)' : '';
      return `- "${field.key}": ${field.description}${optional}`;
    })
    .join('\n');

  return `Return a valid JSON object with the following keys:\n${fieldLines}\nDo not include commentary, Markdown, or code fences. Only output JSON.`;
}

function sanitizeGeminiResponse(text?: string | null) {
  if (!text) {
    return undefined;
  }

  return text.replace(/```json|```/g, '').trim();
}
