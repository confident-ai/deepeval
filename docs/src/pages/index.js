import React from 'react';
import clsx from 'clsx';
import styles from './index.module.css';
import LayoutProvider from '@theme/Layout/Provider';
import Footer from '@theme/Footer';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

class HomeNav extends React.Component {
    render() {
        return <ul className="home-nav">
            <li className=""><Link to={"/docs/getting-started"}>Docs</Link></li>
            <li className=""><Link to="/docs/tutorial-setup">Tutorial</Link></li>
            <li className="header-godoc-link"><a href="https://pkg.go.dev/entgo.io/ent?tab=doc" target="_blank">GoDoc</a></li>
            <li className=""><a href="https://github.com/ent/ent" target="_blank">Github</a></li>
            <li className=""><Link to="/blog/">Blog</Link></li>
        </ul>
    }
}

class Index extends React.Component {
    render() {
      const {config: siteConfig, language = ''} = this.props;
      const {baseUrl} = siteConfig;
  
      const Showcase = () => {
        if ((siteConfig.users || []).length === 0) {
          return null;
        }
  
        const showcase = siteConfig.users
          .filter(user => user.pinned)
          .map(user => (
            <a href={user.infoLink} key={user.infoLink}>
              <img src={user.image} alt={user.caption} title={user.caption} />
            </a>
          ));
  
        const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;
  
        return (
          <div className="productShowcaseSection paddingBottom">
            <h2>Who is Using This?</h2>
            <p>This project is used by all these people</p>
            <div className="logos">{showcase}</div>
            <div className="more-users">
              <a className="button" href={pageUrl('users.html')}>
                More {siteConfig.title} Users
              </a>
            </div>
          </div>
        );
      };
  
      return (
        <div className={"home-splash-container section_index"}>
          <div>ok</div>
        </div>
      );
    }
  }
  

export default function (props) {
    return <LayoutProvider>
        <HomeNav />
        <Index {...props} />
        <Footer/>
    </LayoutProvider>;
};