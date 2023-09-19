import { createSignal } from "solid-js";
import { invoke } from "@tauri-apps/api/tauri";
import styles from './styles.module.css';

function App() {
  return (
  <div class={styles.background}>
    <div class={styles.main}></div>
    <div class={styles.titleback}></div>
    <div class={styles.middlebar}>
      <div class={styles.middlebarrec}></div>
      <div class={styles.middlebartxs}>
        <div class={styles.middlebartxt}>
          <div style="width: 91.36px; left: 0px; top: 0px; position: absolute; text-align: center; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">전체</div>
          <div class="Rectangle7" style="width: 24.14px; height: 3px; left: 34.16px; top: 30px; position: absolute; background: white; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25); border-radius: 2px"></div>
        </div>
        <div class={styles.middlebartxt}>
          <div style="width: 100px; left: 0px; top: 0px; position: absolute; text-align: center; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">동아리</div>-
        </div>
        <div class={styles.middlebartxt} style="width: 120px;">
          <div style="width: 120px; left: 0px; top: 0px; position: absolute; text-align: center; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">앱 / 게임</div>
        </div>
        <div class={styles.middlebartxt}>
          <div style="width: 100px; left: 0px; top: 0px; position: absolute; text-align: center; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">웹 / 서버</div>
        </div>
        <div class={styles.middlebartxt}>
          <div style="width: 100px; left: 0px; top: 0px; position: absolute; text-align: center; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">기타</div>
        </div>
      </div>
      <div class="Search" style="width: 45px; height: 45px; left: 1329px; top: 16px; position: absolute; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25)">
        <div class="Vector" style="width: 45px; height: 45px; left: 0px; top: 0px; position: absolute"></div>
        <div class="Vector" style="width: 32.79px; height: 32.79px; left: 5.62px; top: 5.62px; position: absolute; background: black; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25)"></div>
      </div>
    </div>
    <div class="Title" style="width: 288px; height: 97px; left: 17px; top: 8px; position: absolute">
      <div class="Title" style="width: 288px; height: 93px; left: 0px; top: 4px; position: absolute; background: linear-gradient(106deg, rgba(255, 255, 255, 0.23) 0%, rgba(251.81, 251.81, 251.81, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(15px)"></div>
      <div style="width: 146.94px; height: 33px; left: 109px; top: 31px; position: absolute; color: black; font-size: 32px; font-family: SUITE; font-weight: 700; word-wrap: break-word">선린 스토어</div>
      <img class="12" style="width: 95px; height: 95px; left: 7px; top: 0px; position: absolute" src="https://via.placeholder.com/95x95" />
    </div>
    <div class="Account" style="width: 60px; height: 60px; left: 1331px; top: 125px; position: absolute">
      <div class="Rectangle9" style="width: 60px; height: 60px; left: 0px; top: 0px; position: absolute; background: radial-gradient(46.42% 1.91% at 41.74% 0.97%, rgba(251.81, 251.81, 251.81, 0.49) 0%, rgba(196, 196, 196, 0.09) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(20px)"></div>
      <div class="AccountCircle" style="width: 52.50px; height: 52.50px; left: 3.75px; top: 3.75px; position: absolute">
        <div class="Vector" style="width: 52.50px; height: 52.50px; left: 0px; top: 0px; position: absolute"></div>
        <div class="Vector" style="width: 43.75px; height: 43.75px; left: 4.38px; top: 4.38px; position: absolute; background: linear-gradient(180deg, rgba(0, 0, 0, 0.71) 0%, rgba(0, 0, 0, 0.30) 100%)"></div>
      </div>
    </div>
    <div class="Tags" style="width: 770px; height: 52px; padding-left: 4px; padding-right: 4px; left: 239px; top: 125px; position: absolute; justify-content: flex-start; align-items: center; gap: 18px; display: inline-flex">
      <div class="Window" style="width: 115px; height: 35px; position: relative">
        <div class="Rectangle6" style="width: 115px; height: 35px; left: 0px; top: -0.50px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(255, 255, 255, 0.40) 0%, #C4C4C4 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 5px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="Windows" style="width: 67px; height: 19px; left: 38px; top: 5.50px; position: absolute; color: #565656; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">windows</div>
        <img class="Monitor" style="width: 18px; height: 18px; left: 17px; top: 7.50px; position: absolute" src="https://via.placeholder.com/18x18" />
      </div>
      <div class="Mac" style="width: 87px; height: 35px; position: relative">
        <div class="Rectangle6" style="width: 87px; height: 35px; left: 0px; top: -0.50px; position: absolute; background: radial-gradient(57.93% 3.13% at 69.79% 1.63%, rgba(255, 255, 255, 0.11) 0%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 5px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="Mac" style="width: 67px; height: 19px; left: 38px; top: 5.50px; position: absolute; color: #565656; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">mac</div>
        <img class="MacLogo" style="width: 20px; height: 20px; left: 15px; top: 5.50px; position: absolute" src="https://via.placeholder.com/20x20" />
      </div>
      <div class="Android" style="width: 105px; height: 35px; position: relative">
        <div class="Rectangle6" style="width: 105px; height: 35px; left: 0px; top: -0.50px; position: absolute; background: radial-gradient(57.93% 3.13% at 69.79% 1.63%, rgba(255, 255, 255, 0.11) 0%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 5px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="Android" style="width: 67px; height: 19px; left: 38px; top: 5.50px; position: absolute; color: #565656; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">android</div>
        <img class="AndroidOs" style="width: 20px; height: 20px; left: 15px; top: 5.50px; position: absolute" src="https://via.placeholder.com/20x20" />
      </div>
      <div class="Ios" style="width: 77px; height: 35px; position: relative">
        <div class="Rectangle6" style="width: 77px; height: 35px; left: 0px; top: -0.50px; position: absolute; background: radial-gradient(57.93% 3.13% at 69.79% 1.63%, rgba(255, 255, 255, 0.11) 0%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 5px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="Ios" style="width: 32px; height: 19px; left: 38px; top: 5.50px; position: absolute; color: #565656; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">ios</div>
        <img class="AppleLogo" style="width: 20px; height: 20px; left: 15px; top: 5.50px; position: absolute" src="https://via.placeholder.com/20x20" />
      </div>
      <div class="Web" style="width: 78px; height: 35px; position: relative">
        <div class="Rectangle6" style="width: 78px; height: 35px; left: 0px; top: -0.50px; position: absolute; background: radial-gradient(57.93% 3.13% at 69.79% 1.63%, rgba(255, 255, 255, 0.11) 0%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 5px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="Web" style="width: 35px; height: 19px; left: 33px; top: 6.50px; position: absolute; color: #565656; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">web</div>
        <div class="Web" style="width: 20px; height: 20px; left: 12px; top: 6.50px; position: absolute">
          <div class="Vector" style="width: 20px; height: 20px; left: 0px; top: 0px; position: absolute"></div>
          <div class="Vector" style="width: 16.67px; height: 13.33px; left: 1.67px; top: 3.33px; position: absolute; background: black"></div>
        </div>
      </div>
      <div class="Linux" style="width: 78px; height: 35px; position: relative">
        <div class="Rectangle6" style="width: 78px; height: 35px; left: 0px; top: -0.50px; position: absolute; background: radial-gradient(57.93% 3.13% at 69.79% 1.63%, rgba(255, 255, 255, 0.11) 0%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.25); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="Linux" style="width: 35px; height: 19px; left: 31px; top: 7.50px; position: absolute; color: #565656; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">linux</div>
        <img class="Linux" style="width: 20px; height: 20px; left: 11px; top: 7.50px; position: absolute" src="https://via.placeholder.com/20x20" />
      </div>
    </div>
    <div class="Group16" style="width: 1109px; height: 308px; left: 169px; top: 201px; position: absolute">
      <img class="What1" style="width: 491px; height: 262px; left: 618px; top: 12px; position: absolute; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25); border-radius: 10px" src="https://via.placeholder.com/491x262" />
      <img class="What3" style="width: 491px; height: 262px; left: 0px; top: 12px; position: absolute; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25); border-radius: 10px" src="https://via.placeholder.com/491x262" />
      <img class="What2" style="width: 609px; height: 308px; left: 242px; top: 0px; position: absolute; box-shadow: 0px 4px 4px rgba(0, 0, 0, 0.25); border-radius: 10px; border: 1px black solid" src="https://via.placeholder.com/609x308" />
    </div>
    <div class="Frame18" style="width: 1308px; height: 404px; left: 67px; top: 533px; position: absolute; justify-content: center; align-items: flex-start; gap: 30px; display: inline-flex">
      <div class="Group17" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="Zeropen" style="left: 111px; top: 40px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">ZerOpen</div>
        <div class="5" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">4.5</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Game1" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Game1</div>
      </div>
      <div class="Group18" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div style="left: 111px; top: 40px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">닷지</div>
        <div class="5" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">1.5</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Game2" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Game2</div>
      </div>
      <div class="Group19" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="Rg" style="left: 111px; top: 40px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">RG</div>
        <div class="5" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">3.5</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Game3" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Game3</div>
      </div>
      <div class="Group20" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="ApplePi" style="left: 111px; top: 42px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Apple:Pi</div>
        <div class="0" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">4.0</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="App1" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">App1</div>
      </div>
      <div class="Group21" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div style="left: 111px; top: 43px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">근드캔</div>
        <div class="5" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">3.5</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="App2" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">App2</div>
      </div>
      <div class="Group22" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="Iwop" style="left: 111px; top: 43px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">IWOP</div>
        <div class="5" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">3.5</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Something" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Something</div>
      </div>
      <div class="Group23" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="Ana" style="left: 111px; top: 43px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">AnA</div>
        <div class="0" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">1.0</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Something" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Something</div>
      </div>
      <div class="Group24" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="Ana" style="left: 111px; top: 43px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">AnA</div>
        <div class="0" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">1.0</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Something" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Something</div>
      </div>
      <div class="Group25" style="width: 371px; height: 100px; position: relative">
        <div class="Rectangle6" style="width: 371px; height: 100px; left: 0px; top: 0px; position: absolute; background: radial-gradient(72.03% 8.40% at 28.01% 4.25%, rgba(196, 196, 196, 0.40) 0%, rgba(196, 196, 196, 0.22) 50%, rgba(196, 196, 196, 0) 100%); box-shadow: 0px 10px 10px rgba(0, 0, 0, 0.10); border-radius: 10px; border: 1px rgba(182.75, 182.75, 182.75, 0.20) solid; backdrop-filter: blur(10px)"></div>
        <div class="StarRate" style="width: 25px; height: 25px; left: 136.75px; top: 62px; position: absolute">
          <div class="Group" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute">
            <div class="Vector" style="width: 25px; height: 25px; left: 0px; top: 0px; position: absolute"></div>
            <div class="Vector" style="width: 20.83px; height: 20.83px; left: 2.08px; top: 2.08px; position: absolute; background: black"></div>
          </div>
        </div>
        <div class="Ana" style="left: 111px; top: 43px; position: absolute; color: #818181; font-size: 16px; font-family: SUITE; font-weight: 600; word-wrap: break-word">AnA</div>
        <div class="0" style="width: 33px; left: 104px; top: 67px; position: absolute; text-align: center; color: black; font-size: 16px; font-family: SUITE; font-weight: 700; word-wrap: break-word">1.0</div>
        <div class="Rectangle11" style="width: 80px; height: 80px; left: 15px; top: 10px; position: absolute; background: black; border-radius: 30px"></div>
        <div class="Something" style="width: 183px; left: 104px; top: 16px; position: absolute; color: black; font-size: 24px; font-family: SUITE; font-weight: 600; word-wrap: break-word">Something</div>
      </div>
    </div>
  </div>
  );
}

export default App;
