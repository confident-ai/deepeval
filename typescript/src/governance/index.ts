import { Api, Endpoints, HttpMethods } from "../confident/api";

export interface GovernancePolicyReference {
  id: string;
  name: string;
}

export interface GovernanceAssessment {
  passed: boolean;
  governancePolicy: GovernancePolicyReference;
}

export async function assessGovernance(
  apiKey?: string,
): Promise<GovernanceAssessment> {
  const api = new Api(apiKey);
  const response = await api.sendRequest(
    HttpMethods.POST,
    Endpoints.GOVERNANCE_ASSESS_ENDPOINT,
  );
  return response.data as GovernanceAssessment;
}
