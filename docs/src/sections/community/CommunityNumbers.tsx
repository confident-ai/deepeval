import { communityEvents, countCities } from "@/lib/community/events";
import { contributionCount, contributorCount } from "@/lib/community/contributors";
import CommunityStats from "./CommunityStats";

// Server wrapper: reads the build data, so the client tile bundle stays small.
const CommunityNumbers: React.FC = () => (
  <CommunityStats
    contributors={contributorCount}
    contributions={contributionCount}
    events={communityEvents.length}
    cities={countCities(communityEvents)}
  />
);

export default CommunityNumbers;
