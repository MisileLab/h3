import { Line } from "./Elements";
import { styled } from "solid-styled-components";
import para_image from "../assets/media/PARA.jpg";

const Container = styled.div`
  width: 100%;
  height: 110vh;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const TitleText = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: center;
  font-style: normal;
  font-weight: 600;
  font-size: 36px;
  line-height: 120%;
  @media (max-width: 800px) {
    font-size: 26px;
  }
`;

const BigWanjangContainer = styled.div`
  display: inline-flex;
  justify-content: center;
  align-items: center;
  gap: 50px;
  margin-top: 40px;
  @media (max-width: 800px) {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 40px;
  }
`;

const PeriodText = styled.div`
  color: #000;
  text-align: center;
  font-size: 28px;
  font-style: normal;
  font-weight: 500;
  line-height: 170%;
  margin-top: 40px;
  @media (max-width: 800px) {
    font-size: 18px;
  }
`;

// 부장 쀼장 이름 텍스트
const WanjanNameText = styled.div`
  text-align: center;
  font-style: normal;
  font-weight: 600;
  font-size: 26px;
  line-height: 120%;
  @media (max-width: 800px) {
    font-size: 18px;
  }
`;

const WanjangContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: 400px;
  @media (max-width: 800px) {
    width: 300px;
  }
`;

const Text = styled.div`
  color: #000;
  text-align: center;
  font-size: 20px;
  font-style: normal;
  font-weight: 400;
  line-height: 140%; /* 39.2px */
  @media (max-width: 800px) {
    font-size: 14px;
  }
`;

const CareerText = styled.div`
  color: #000;
  font-size: 20px;
  font-style: normal;
  font-weight: 400;
  line-height: 140%; /* 39.2px */
`;

const Image = styled.div`
  width: 120px;
  height: 120px;
  border: solid 1px;
  border-color: black;
  border-radius: 10px;
  display: flex;
  background-image: url(${para_image});
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  @media (max-width: 800px) {
    display: none;
  }
`;

const Button = styled.button`
  display: inline-flex;
  width: 300px;
  height: 60px;
  justify-content: center;
  align-items: center;
  gap: 10px;
  border-radius: 15px;
  background: rgba(255, 255, 255, 0);
  border-color: rgba(0, 0, 0, 0);
  border: solid 1px;
  transition: background-color 0.2s ease-in-out;

  &:hover {
    background: rgba(170, 170, 170, 0.5);
  }

  @media (max-width: 800px) {
    width: 200px;
    height: 40px;
    border-radius: 10px;
  }
`;

const JiwonButton = styled.button`
  display: inline-flex;
  width: 300px;
  height: 60px;
  justify-content: center;
  align-items: center;
  gap: 10px;
  border-radius: 15px;
  background: #440c49;
  box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.25);
  margin-top: 20px;
  border: none;

  &:hover {
    background: #4f0d67;
  }
  @media (max-width: 800px) {
    width: 200px;
    height: 40px;
    border-radius: 10px;
  }
`;

const ButtonText = styled.div`
  color: #fff;
  text-align: center;
  font-size: 18px;
  font-style: normal;
  font-weight: 500;
  line-height: 140%; /* 30.8px */
  @media (max-width: 800px) {
    font-size: 14px;
  }
`;

const Jiwon = () => {
  let isJiwonGigan = true; //현재 지원 기간인가?
  let isJiwon = false; //지원이 가능한가?
  let timeDiff = 0;

  let days = 0;
  let hours = 0;
  let minutes = 0;
  let seconds = 0;

  const startTime = new Date("2023-12-18T18:00:00");
  const endTime = new Date("2023-12-24T23:59:59");
  const toParaInsta = () => {
    window.open(
      "https://www.instagram.com/sunrin_para/",
      "_blank",
      "noreferrer",
    );
  };

  const toJiwonForm = () => {
    if (isJiwon) {
      window.open(
        "https://forms.gle/cVMum1hGpMB3gNPf6",
        "_blank",
        "noreferrer",
      );
    }
  };

  const TimeUpdate = () => {
    let currentTime = new Date();

    let period = document.getElementById("Period");
    if (period == null) return;

    let result = "";
    if (
      !isJiwonGigan ||
      startTime.getFullYear() !== currentTime.getFullYear() ||
      endTime.getDate() < currentTime.getDate()
    ) {
      result = "지원 기간이 아닙니다.";
      clearInterval(timer);
    } else {
      timeDiff = startTime.getTime() - currentTime.getTime();
      days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
      hours = Math.floor((timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
      seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);

      if (timeDiff < 0) {
        isJiwon = true;
        timeDiff = endTime.getTime() - currentTime.getTime();
        days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
        hours = Math.floor(
          (timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60),
        );
        minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
        seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);

        result = result = `지원 기간 : 12월 18일 18시 ~ 12월 24일<br>
                지원 마감까지 남은 시간 : ${days}일 ${hours}시간 ${minutes}분 ${seconds}초`;
      } else {
        result = `지원 기간 : 12월 18일 18시 ~ 12월 24일<br>
                지원 시작까지 남은 시간 : ${days}일 ${hours}시간 ${minutes}분 ${seconds}초`;
      }
    }

    period.innerHTML = result;
  };
  let timer = setInterval(TimeUpdate, 1000);

  return (
    <Container>
      <TitleText> 지원 안내 및 문의 </TitleText>
      <PeriodText id={"Period"}> 지원 기간 : 추후 공개 예정</PeriodText>
      <BigWanjangContainer>
        <WanjangContainer>
          <Image />
          <WanjanNameText>동아리 문의 연락처</WanjanNameText>
          <Text>
            Insta : @sunrin_para
            <br />
            facebook : 미개설
          </Text>
        </WanjangContainer>
        <Line />
        <WanjangContainer>
          <WanjanNameText>부장 이정훈</WanjanNameText>
          <Text>Insta : @compy07</Text>
          <WanjanNameText style={{ "margin-top": "30px" }}>
            쀼장 유채호
          </WanjanNameText>
          <Text>
            Insta : @chaeho_yu_&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Tell :
            01087343741
          </Text>
          <CareerText></CareerText>
        </WanjangContainer>
      </BigWanjangContainer>

      <Button onClick={toParaInsta} style={{ "margin-top": "40px" }}>
        <ButtonText style={{ color: "black" }}>인스타그램으로 문의</ButtonText>
      </Button>

      <JiwonButton onClick={toJiwonForm}>
        <ButtonText>PARA 지원하기</ButtonText>
      </JiwonButton>
    </Container>
  );
};

export default Jiwon;
