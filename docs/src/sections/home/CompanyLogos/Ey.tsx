import type { LogoProps } from "./types";
import styles from "./CompanyLogos.module.scss";

const Ey: React.FC<LogoProps> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 68.67 69.32" {...props}>
    <path
      className={styles.themedDark}
      d="M11.09 61.4h17.37v7.92H.67V34.9h19.7l4.61 7.92H11.1v5.68h12.56v7.22H11.1zm35.86-26.5l-5.9 11.23-5.88-11.23H23.65l12.13 20.82v13.6h10.4v-13.6L58.31 34.9z"
      fill="#161d23"
      fillRule="evenodd"
    />
    <path
      fill="#ffe600"
      fillRule="evenodd"
      d="M68.67 12.81V0L0 24.83z"
    />
  </svg>
);

export default Ey;
