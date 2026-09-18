/**
 * Typed API endpoints matching 08-API-SPECIFICATION.md
 */

import { apiClient } from './client';
import {
  AnalysisCreateResponse,
  AnalysisListResponse,
  AnalysisPollResponse,
  AnalysisResponse,
  ChallengeIssueResponse,
  ChallengeVerifyResponse,
  HealthResponse,
  SystemMetrics,
} from '../types/api';

export const api = {
  // Auth
  auth: {
    login: async (formData: FormData) => {
      const res = await apiClient.post('/auth/login', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    },
    register: async (payload: { email: string; password: string; display_name?: string }) => {
      const res = await apiClient.post('/auth/register', payload);
      return res.data;
    },
    me: async () => {
      const res = await apiClient.get('/auth/me');
      return res.data;
    },
    logout: async () => {
      const res = await apiClient.post('/auth/logout');
      return res.data;
    },
  },

  // Analyses
  analyses: {
    create: async (formData: FormData): Promise<AnalysisCreateResponse> => {
      const res = await apiClient.post('/analyses', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    },
    poll: async (id: string): Promise<AnalysisPollResponse> => {
      const res = await apiClient.get(`/analyses/${id}`);
      return res.data;
    },
    get: async (id: string): Promise<AnalysisResponse> => {
      const res = await apiClient.get(`/analyses/${id}`);
      return res.data;
    },
    list: async (params?: {
      page?: number;
      page_size?: number;
      verdict?: string;
    }): Promise<AnalysisListResponse> => {
      const res = await apiClient.get('/analyses', { params });
      return res.data;
    },
    delete: async (id: string): Promise<{ status: string; id: string }> => {
      const res = await apiClient.delete(`/analyses/${id}`);
      return res.data;
    },
  },

  // Challenge
  challenge: {
    issue: async (analysisId: string, challengeType?: string): Promise<ChallengeIssueResponse> => {
      const res = await apiClient.post(`/analyses/${analysisId}/challenge`, {
        challenge_type: challengeType,
      });
      return res.data;
    },
    verify: async (
      analysisId: string,
      challengeId: string,
      audioBlob: Blob,
      filename = 'response.wav'
    ): Promise<ChallengeVerifyResponse> => {
      const formData = new FormData();
      formData.append('audio', audioBlob, filename);
      const res = await apiClient.post(
        `/analyses/${analysisId}/challenge/${challengeId}/response`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );
      return res.data;
    },
  },

  // Artifacts
  artifacts: {
    getUrl: (_analysisId: string, artifactId: string) => {
      return `/api/v1/artifacts/${artifactId}`;
    },
  },

  // System
  system: {
    health: async (): Promise<HealthResponse> => {
      const res = await apiClient.get('/system/health');
      return res.data;
    },
    metrics: async (): Promise<SystemMetrics> => {
      const res = await apiClient.get('/system/metrics');
      return res.data;
    },
  },
};
