import contributors from "@/lib/generated/repo-contributors.json";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import styles from "./ContributorMarquee.module.scss";

type Contributor = { login: string; avatarUrl: string; url: string; contributions: number };

// Two rows of contributor avatars scrolling in opposite directions along
// the bottom of the hero. Avatars are fetched at 12px and shown at 24px
// with pixelated scaling, so they read as pixel portraits next to the
// mascots. Hovering one outlines it in purple.
const PER_ROW = 50;

const avatar = (url: string) => `${url}${url.includes("?") ? "&" : "?"}s=12`;

function Row({ people, reverse }: { people: Contributor[]; reverse?: boolean }) {
  return (
    <div className={styles.row}>
      <div className={styles.track} data-reverse={reverse || undefined}>
        {[...people, ...people].map((c, i) => (
          <a
            key={`${c.login}-${i}`}
            className={styles.face}
            href={c.url}
            target="_blank"
            rel="noopener noreferrer"
            title={`@${c.login}`}
            aria-hidden={i >= people.length || undefined}
            tabIndex={i >= people.length ? -1 : undefined}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={avatar(c.avatarUrl)} alt={`@${c.login}`} width={24} height={24} loading="lazy" />
          </a>
        ))}
      </div>
    </div>
  );
}

const ContributorMarquee: React.FC = () => {
  const list = (contributors as Contributor[]).filter((c) => !c.login.endsWith("[bot]"));
  if (list.length === 0) return null;
  const top = list.slice(0, PER_ROW * 2);
  const rowA = top.filter((_, i) => i % 2 === 0);
  const rowB = top.filter((_, i) => i % 2 === 1);
  return (
    <PauseOffscreen className={styles.marquee} aria-label={`${list.length} contributors build DeepEval`}>
      <Row people={rowA} />
      <Row people={rowB} reverse />
    </PauseOffscreen>
  );
};

export default ContributorMarquee;
