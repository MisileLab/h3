import { styled } from "solid-styled-components";

const Container = styled.div`
  width: 100%;
  height: 100vh;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const EssentialContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 120px;
  @media (max-width: 1100px) {
    gap: 0px;
  }
`;

const TitleText = styled.div`
  color: #000;
  text-align: right;
  font-size: 36px;
  font-style: normal;
  font-weight: 600;
  line-height: normal;
  width: 266px;
  height: 215px;
  @media (max-width: 1100px) {
    display: none;
  }
`;

const DetailTitle = styled.div`
  width: 700px;
  color: #000;
  font-size: 22px;
  font-style: normal;
  font-weight: 600;
  line-height: 140%; /* 39.2px */
  @media (max-width: 800px) {
    width: 85vw;
    font-size: 18px;
  }
`;

const DetailText = styled.div`
  width: 700px;
  color: #000;
  font-size: 22px;
  font-style: normal;
  font-weight: 400;
  line-height: 140%; /* 39.2px */
  @media (max-width: 800px) {
    width: 85vw;
    font-size: 16px;
  }
`;

const DetailContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 20px;
  @media (max-width: 800px) {
    width: 85vw;
    gap: 15px;
  }
`;

const Essential = () => {
  return (
    <Container>
      <EssentialContainer>
        <TitleText>
          인공지능에 대하여
          <br />
          연구하고,
          <br />
          개발하고,
          <br />
          성취하는,
          <br />
          동아리.
        </TitleText>
        <DetailContainer>
          <DetailTitle>
            PARA는 2024학년도 첫 활동을 시작하는
            <br />
            선린인터넷고의 인공지능 연구 및 개발 동아리입니다.
          </DetailTitle>
          <DetailText>
            오픈소스로 공개된 인공지능 모델을 학습 또는 튜닝을 진행하여 실제
            서비스 제작에 응용할 수 있는 능력을 기르는데 중점을 두고 있습니다.
          </DetailText>
          <DetailText>
            인공지능에 대한 높은 이해도와 관심을 가지고 있는 실력자들과 인공지능
            이외에도 각종 개발 분야의 특기를 가진 능력자들로 구성되어있습니다.
          </DetailText>
          <DetailText>
            수업 커리큘럼 등의 운영 계획을 심혈을 기울여 준비하였습니다.
            <br />
            전공 동아리 못지 않은, 그 이상의 것들로 준비하였습니다.
          </DetailText>
        </DetailContainer>
      </EssentialContainer>
    </Container>
  );
};

export default Essential;
