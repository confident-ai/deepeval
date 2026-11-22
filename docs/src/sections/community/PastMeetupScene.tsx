import styles from "./PastMeetupScene.module.scss";

// The SF meetup as pixel art: the photo of the room (from under the
// ceiling to the front chairs), drawn at 240 x 180 in eight greys, with a
// dithered border that thins out toward every edge. The dark theme gets a
// version with lifted shadows so dark clothes still read. File names carry
// a version, so a new render is never hidden by a cached old one.
const SCENE = {
  light: "/community/2026-11-20-san-francisco-room-v5.png",
  dark: "/community/2026-11-20-san-francisco-room-v5-dark.png",
  alt: "Pixel art of the DeepEval SF meetup: the speaker at the podium beside the slide, and the audience in rows of chairs, some on laptops.",
};

const PastMeetupScene: React.FC = () => (
  <figure className={`not-prose ${styles.scene}`}>
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img className={styles.light} src={SCENE.light} alt={SCENE.alt} loading="lazy" />
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img className={styles.dark} src={SCENE.dark} alt="" aria-hidden loading="lazy" />
    <figcaption className={styles.caption}>
      🇺🇸 San Francisco · Nov 20, 2026 · our first meetup
    </figcaption>
  </figure>
);

export default PastMeetupScene;
