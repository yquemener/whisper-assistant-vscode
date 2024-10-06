import * as vscode from 'vscode';
import { exec, ChildProcess } from 'child_process';
import * as fs from 'fs';
import { promisify } from 'util';
import * as path from 'path';

const execAsync = promisify(exec);

interface Segment {
  id: number;
  seek: number;
  start: number;
  end: number;
  text: string;
  tokens: number[];
  temperature: number;
}

export interface Transcription {
  text: string;
  segments: Segment[];
  language: string;
}

export type WhisperModel = 'tiny' | 'base' | 'small' | 'medium' | 'large';

export default class SpeechTranscription {
  private outputDir: string;
  private recordingProcess: ChildProcess | null = null;
  private tempFileName: string = 'recording_temp.wav';
  private finalFileName: string = 'recording.wav';
  private outputChannel: vscode.OutputChannel;
  private currentRecordingId: string = '';

  constructor(
    outputDir: string,
    outputChannel: vscode.OutputChannel,
  ) {
    this.outputDir = outputDir;
    this.outputChannel = outputChannel;
  }

  async checkIfInstalled(command: string): Promise<boolean> {
    try {
      await execAsync(`${command} --help`);
      return true;
    } catch (error) {
      return false;
    }
  }

  getOutputDir(): string {
    return this.outputDir;
  }

  startRecording(): void {
    this.currentRecordingId = Date.now().toString();
    const tempFilePath = path.join(this.outputDir, `${this.currentRecordingId}_${this.tempFileName}`);
    this.recordingProcess = exec(
      `sox -d -b 16 -e signed -c 1 -r 16k ${tempFilePath}`,
      (error, stdout, stderr) => {
        if (error) {
          this.outputChannel.appendLine(`Whisper Assistant: error: ${error}`);
          return;
        }
        if (stderr) {
          this.outputChannel.appendLine(
            `Whisper Assistant: SoX process has been killed: ${stderr}`,
          );
          return;
        }
        this.outputChannel.appendLine(`Whisper Assistant: stdout: ${stdout}`);
      },
    );
  }

  async stopRecording(): Promise<void> {
    if (!this.recordingProcess) {
      this.outputChannel.appendLine(
        'Whisper Assistant: No recording process found',
      );
      return;
    }
    this.outputChannel.appendLine('Whisper Assistant: Stopping recording');
    this.recordingProcess.kill();
    this.recordingProcess = null;

    // Rename the file to trigger transcription
    const tempFilePath = path.join(this.outputDir, `${this.currentRecordingId}_${this.tempFileName}`);
    const finalFilePath = path.join(this.outputDir, `${this.currentRecordingId}_${this.finalFileName}`);
    await fs.promises.rename(tempFilePath, finalFilePath);
    this.outputChannel.appendLine('Whisper Assistant: Recording saved and ready for transcription');
  }

  public async transcribeRecording(model: WhisperModel, language: string): Promise<Transcription | undefined> {
    try {
      const config = vscode.workspace.getConfiguration('whisperAssistant');
      const language = config.get<string>('transcriptionLanguage', 'en');

      this.outputChannel.appendLine(
        `Whisper Assistant: Transcribing recording using '${model}' model and '${language}' language`,
      );

      const audioFilePath = path.join(this.outputDir, `${this.currentRecordingId}_${this.finalFileName}`);
      
      // Use the transcribe method
      const transcription = await this.transcribe(audioFilePath);

      this.outputChannel.appendLine(
        `Whisper Assistant: Transcription completed`,
      );

      return transcription;
    } catch (error) {
      this.outputChannel.appendLine(`Whisper Assistant: error: ${error}`);
    }
  }

  async transcribe(audioFilePath: string): Promise<Transcription> {
    const baseName = path.basename(audioFilePath, '.wav');
    const resultPath = path.join(this.outputDir, `${baseName}.json`);
    this.outputChannel.appendLine(`Whisper Assistant: Waiting for result file: ${resultPath}`);
    
    // Wait for the transcription result
    await this.waitForFile(resultPath);
    
    // Read and parse the result
    const resultJson = await fs.promises.readFile(resultPath, 'utf-8');
    const result = JSON.parse(resultJson);
    
    // Clean up both WAV and JSON files
    await fs.promises.unlink(audioFilePath);
    await fs.promises.unlink(resultPath);
    
    this.outputChannel.appendLine(`Whisper Assistant: Result file found and read: ${resultPath}`);
    return result;
  }

  private async waitForFile(filePath: string, timeout = 30000): Promise<void> {
    const startTime = Date.now();
    while (true) {
      if (await fs.promises.access(filePath).then(() => true).catch(() => false)) {
        // Add a small delay to ensure the file is fully written
        await new Promise(resolve => setTimeout(resolve, 100));
        return;
      }
      if (Date.now() - startTime > timeout) {
        throw new Error(`Timeout waiting for transcription result: ${filePath}`);
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  deleteFiles(): void {
    // Delete the WAV file
    const finalFilePath = path.join(this.outputDir, this.finalFileName);
    if (fs.existsSync(finalFilePath)) {
      fs.unlinkSync(finalFilePath);
    }
  }
}