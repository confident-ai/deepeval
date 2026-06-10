import { ConversationalGolden, Golden } from "./golden";

export interface APIDataset {
  overwrite?: boolean;
  version?: string;
  goldens?: Golden[];
  conversationalGoldens?: any[];
}

export interface CreateDatasetHttpResponse {
  link: string;
}

export interface DatasetHttpResponse {
  goldens: Golden[];
  conversationalGoldens: ConversationalGolden[];
  id: string;
  version?: string | null;
}

export interface DatasetVersion {
  id: string;
  version: string;
  createdAt?: string;
}

export interface GetDatasetVersionsResponse {
  versions: DatasetVersion[];
}

export interface CreateDatasetVersionResponse {
  id: string;
  version: string;
}
