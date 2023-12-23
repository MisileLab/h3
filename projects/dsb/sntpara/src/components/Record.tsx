import { Line } from "./Elements";
import { styled } from "solid-styled-components";

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
  @media (max-width: 800px) {
    font-size: 14px;
  }
`;

const Record = () => {
  return (
    <Container>
      <TitleText> 동아리 실적 </TitleText>
      <BigWanjangContainer>
        <WanjangContainer>
          <WanjanNameText>부장 이정훈</WanjanNameText>
          <CareerText>
            - 소프트웨어과 118기 특별교육이수자전형 입학
            <br />
            - 2023 한국 코드페어 SW 공모전 본선
            <br />
            - 직업계고 게임개발대회 장려상
            <br />
            - 정보올림피아드 장려상
            <br />- 현 제로픈 20기
          </CareerText>
        </WanjangContainer>
        <Line />
        <WanjangContainer>
          <WanjanNameText>쀼장 유채호</WanjanNameText>
          <CareerText>
            - 2020 카이스트 영재원 C언어반
            <br />
            - 2021 한양대 SW 영재원 심화과정, 우수상
            <br />
            - 소프트웨어과 118기 미래인재전형 입학
            <br />
            - 2023 선린 해커톤 게임 부문 대상
            <br />- 현 RG 23기
          </CareerText>
        </WanjangContainer>
      </BigWanjangContainer>
    </Container>
  );
};

export default Record;
