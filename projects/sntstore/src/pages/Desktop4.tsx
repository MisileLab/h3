import { FunctionComponent } from "react";
import styles from "./Desktop4.module.css";

const Desktop4: FunctionComponent = () => {
  return (
    <div className={styles.desktop4}>
      <div className={styles.middleBar}>
        <div className={styles.middleBarChild} />
        <div className={styles.groupParent}>
          <div className={styles.parent}>
            <div className={styles.div}>전체</div>
            <div className={styles.groupChild} />
          </div>
          <div className={styles.group}>
            <div className={styles.div1}>동아리</div>
            <div className={styles.groupItem} />
          </div>
          <div className={styles.container}>
            <div className={styles.div2}>앱 / 게임</div>
            <div className={styles.groupInner} />
          </div>
          <div className={styles.group}>
            <div className={styles.div1}>웹 / 서버</div>
            <div className={styles.groupItem} />
          </div>
          <div className={styles.group}>
            <div className={styles.div1}>기타</div>
            <div className={styles.groupItem} />
          </div>
        </div>
        <img className={styles.searchIcon} alt="" src="/search.svg" />
        <img className={styles.icon} alt="" src="/-1-2@2x.png" />
        <b className={styles.b}>선린 스토어</b>
      </div>
      <img className={styles.accountIcon} alt="" src="/account.svg" />
      <div className={styles.tags}>
        <div className={styles.window}>
          <div className={styles.windowChild} />
          <div className={styles.windows}>windows</div>
          <img className={styles.monitorIcon} alt="" src="/monitor@2x.png" />
        </div>
        <div className={styles.mac}>
          <div className={styles.macChild} />
          <div className={styles.windows}>mac</div>
          <img className={styles.macLogoIcon} alt="" src="/mac-logo@2x.png" />
        </div>
        <div className={styles.android}>
          <div className={styles.androidChild} />
          <div className={styles.windows}>android</div>
          <img className={styles.macLogoIcon} alt="" src="/android-os@2x.png" />
        </div>
        <div className={styles.ios}>
          <div className={styles.iosChild} />
          <div className={styles.ios1}>ios</div>
          <img className={styles.macLogoIcon} alt="" src="/apple-logo@2x.png" />
        </div>
        <div className={styles.web}>
          <div className={styles.webChild} />
          <div className={styles.web1}>web</div>
          <img className={styles.webIcon} alt="" src="/web.svg" />
        </div>
        <div className={styles.web}>
          <div className={styles.webChild} />
          <div className={styles.linux1}>linux</div>
          <img className={styles.linuxIcon} alt="" src="/linux@2x.png" />
        </div>
      </div>
      <div className={styles.what1Parent}>
        <img className={styles.what1Icon} alt="" src="/what-1@2x.png" />
        <img className={styles.what3Icon} alt="" src="/what-3@2x.png" />
        <img className={styles.what2Icon} alt="" src="/what-2@2x.png" />
      </div>
      <div className={styles.groupContainer}>
        <div className={styles.rectangleParent}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate.svg" />
          <div className={styles.zeropen}>ZerOpen</div>
          <b className={styles.b1}>4.5</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Game1</div>
        </div>
        <div className={styles.rectangleParent}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate1.svg" />
          <div className={styles.zeropen}>닷지</div>
          <b className={styles.b1}>1.5</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Game2</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate1.svg" />
          <div className={styles.zeropen}>RG</div>
          <b className={styles.b1}>3.5</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Game3</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild8} />
          <img className={styles.starRateIcon} alt="" src="/star-rate.svg" />
          <div className={styles.applepi}>Apple:Pi</div>
          <b className={styles.b1}>4.0</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>App1</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate1.svg" />
          <div className={styles.iwop}>근드캔</div>
          <b className={styles.b1}>3.5</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>App2</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate1.svg" />
          <div className={styles.iwop}>IWOP</div>
          <b className={styles.b1}>3.5</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Something</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate.svg" />
          <div className={styles.iwop}>AnA</div>
          <b className={styles.b1}>1.0</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Something</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate1.svg" />
          <div className={styles.iwop}>AnA</div>
          <b className={styles.b1}>1.0</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Something</div>
        </div>
        <div className={styles.rectangleContainer}>
          <div className={styles.groupChild2} />
          <img className={styles.starRateIcon} alt="" src="/star-rate1.svg" />
          <div className={styles.iwop}>AnA</div>
          <b className={styles.b1}>1.0</b>
          <div className={styles.groupChild3} />
          <div className={styles.something}>Something</div>
        </div>
      </div>
    </div>
  );
};

export default Desktop4;
