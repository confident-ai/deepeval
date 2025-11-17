import React from "react";
import styles from "./index.module.css";
import LayoutProvider from "@theme/Layout/Provider";
import Footer from "@theme/Footer";
import Link from "@docusaurus/Link";

class HomeNav extends React.Component {
  render() {
    return (
      <div className={styles.homeNav}>
        <div>
          <Link to={"/docs/getting-started"}>Docs</Link>
        </div>
        <div>
          <Link to="https://www.confident-ai.com/docs">Confident AI</Link>
        </div>
        <div>
          <Link to="/guides/guides-rag-evaluation">Guides</Link>
        </div>
        <div>
          <Link to="/tutorials/tutorial-introduction">Tutorials</Link>
        </div>
        <div className={styles.canHide}>
          <div>
            <a href="https://github.com/confident-ai/deepeval" target="_blank">
              Github
            </a>
          </div>
          <div>
            <a href="https://confident-ai.com/blog" target="_blank">
              Blog
            </a>
          </div>
        </div>
      </div>
    );
  }
}

class ConfidentEnvelope extends React.Component {
  handleConfident = () => {
    window.open("https://confident-ai.com", "_blank");
  };

  render() {
    return (
      <div className={styles.letterContainer} onClick={this.handleConfident}>
        <div className={styles.letterImage}>
          <div className={styles.animatedMail}>
            <div className={styles.backFold}></div>
            <div className={styles.letter}>
              <div className={styles.letterBorder}></div>
              <div className={styles.letterTitle}>Delivered by</div>
              <div className={styles.letterContentContainer}>
                <img
                  src="icons/logo.svg"
                  style={{ width: "30px", height: "30px" }}
                />
                <div className={styles.letterContext}>
                  <span class="lexend-deca" style={{ fontSize: "16px" }}>
                    Confident AI
                  </span>
                </div>
              </div>
              <div className={styles.letterStamp}>
                <div className={styles.letterStampInner}></div>
              </div>
            </div>
            <div className={styles.topFold}></div>
            <div className={styles.body}></div>
            <div className={styles.leftFold}></div>
          </div>
          <div className={styles.shadow}></div>
        </div>
      </div>
    );
  }
}

class FeatureCard extends React.Component {
  render() {
    const { title, link, description } = this.props;

    return (
      <Link to={link} className={styles.featureCard}>
        <div className={styles.featureCardContainer}>
          <span className={styles.title}>
            {title}
            <img src="icons/right-arrow.svg" />
          </span>
        </div>
        <p className={styles.description}>{description}</p>
      </Link>
    );
  }
}

class Index extends React.Component {
  handleConfident = () => {
    window.open("https://confident-ai.com", "_blank");
  };

  render() {
    return (
      <div className={styles.mainMainContainer}>
        <div className={styles.mainContainer}>
          <div className={styles.mainLeftContainer}>
            <img src="icons/DeepEval.svg" />
            <div className={styles.contentContainer}>
              <h1>$ the open-source LLM evaluation framework</h1>
              <div className={styles.ctaContainer}>
                <Link to={"/docs/getting-started"} className={styles.button}>
                  Get Started
                </Link>
                <a
                  href={"https://confident-ai.com"}
                  className={styles.confidentLink}
                >
                  <span>Try the DeepEval Platform</span>
                  <img className={styles.newTabIcon} src="icons/new-tab.svg" />
                </a>
              </div>
            </div>
          </div>
          <ConfidentEnvelope />
        </div>
        <div className={styles.featuresContainer}>
          <FeatureCard
            title="Unit-Testing for LLMs"
            link="/docs/evaluation-end-to-end-llm-evals#use-deepeval-test-run-in-cicd-pipelines"
            description="Native integration with Pytest, that fits right in your workflow."
          />
          <FeatureCard
            title="LLM-as-a-Judge Metrics"
            link="/docs/metrics-introduction"
            description="50+ research-backed metrics, including custom G-Eval and deterministic metrics."
          />
          <FeatureCard
            title="Single and Multi-Turn Evals"
            link="/docs/evaluation-test-cases"
            description="Covering any use cases, any system architecture, including multi-modality."
          />
        </div>
      </div>
    );
  }
}

export default function (props) {
  return (
    <LayoutProvider>
      <div className={styles.mainRapper}>
        <div className={styles.rapper}>
          <HomeNav />
          <Index {...props} />
        </div>
      </div>
      <Footer />
    </LayoutProvider>
  );
}
