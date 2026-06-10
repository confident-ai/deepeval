export enum AnnotationType {
  THUMBS_RATING = "THUMBS_RATING",
  FIVE_STAR_RATING = "FIVE_STAR_RATING",
}

export interface APIAnnotation {
  rating: number;
  traceUuid?: string;
  spanUuid?: string;
  threadId?: string;
  expectedOutput?: string;
  expectedOutcome?: string;
  explanation?: string;
  type?: AnnotationType;
  userId?: string;
}
