// Runs in <head> before first paint (rendered in app/layout.tsx). Adds the
// nd-banner-<id> class for every bar the visitor closed, so a closed
// AnnouncementBar (lib/announcement-bar.tsx) never flashes. Add the id of
// every new bar here.
const BANNER_IDS = ["sf-meetup", "docs-announcement"];

export const bannerRestoreScript = `try{for(const id of ${JSON.stringify(BANNER_IDS)})if(localStorage.getItem("nd-banner-"+id)==="true")document.documentElement.classList.add("nd-banner-"+id)}catch(e){}`;
