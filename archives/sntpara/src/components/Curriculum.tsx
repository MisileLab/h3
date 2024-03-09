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

const TitleText = styled.div`
  color: #000;
  text-align: center;
  font-size: 36px;
  font-style: normal;
  font-weight: 600;
  line-height: 120%; /* 43.2px */
  @media (max-width: 800px) {
    font-size: 26px;
  }
`;

const BigContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 70px;
  @media (max-width: 800px) {
    gap: 50px;
  }
`;

const CurriculumContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 40px;
  @media (max-width: 800px) {
    gap: 30px;
  }
`;

const CurriculumGroup = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 30px;
  @media (max-width: 800px) {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 20px;
  }
`;

const CurriculumBox = styled.div`
  display: flex;
  width: 500px;
  flex-direction: column;
  align-items: center;
  gap: 15px;
  @media (max-width: 800px) {
    gap: 10px;
    width: 300px;
  }
`;

const CurriculumTitle = styled.div`
  color: #000;
  font-size: 30px;
  font-style: normal;
  font-weight: 600;
  line-height: 120%; /* 36px */
  @media (max-width: 800px) {
    font-size: 18px;
  }
`;

const CurriculumDetailText = styled.div`
  color: #000;
  font-size: 22px;
  font-style: normal;
  font-weight: 400;
  line-height: 120%; /* 26.4px */
  @media (max-width: 800px) {
    font-size: 14px;
  }
`;

const Curriculum = () => {
  return (
    <Container>
      <BigContainer>
        <TitleText>커리큘럼 및 운영계획</TitleText>
        <CurriculumContainer>
          <CurriculumGroup>
            <CurriculumBox>
              <CurriculumTitle>1학기</CurriculumTitle>
              <CurriculumDetailText>
                {" "}
                - 파이썬 문법, 프로그래밍 기초 이론 수업 - 1달
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 언어모델의 원리와 파인튜닝 수업 - 2달
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - OpenCV를 사용한 이미지 처리와 학습 수업 - 1달
              </CurriculumDetailText>
            </CurriculumBox>
            <CurriculumBox>
              <CurriculumTitle>2학기</CurriculumTitle>
              <CurriculumDetailText>
                {" "}
                - 선린에서 데이터 조사 및 수집, 정보 윤리 수업 - 1달{" "}
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 수집한 데이터를 활용하여 머신러닝 수업 - 2달
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 딥러닝 기초 이론 수업 - 1달{" "}
              </CurriculumDetailText>
            </CurriculumBox>
          </CurriculumGroup>
          <CurriculumGroup>
            <CurriculumBox>
              <CurriculumTitle>여름방학</CurriculumTitle>
              <CurriculumDetailText>
                {" "}
                - 파이썬 백엔드 프레임워크를 사용한 REST API 특강
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 파이썬을 이용한 자유 연구 프로젝트
              </CurriculumDetailText>
            </CurriculumBox>
            <CurriculumBox>
              <CurriculumTitle>겨울방학</CurriculumTitle>
              <CurriculumDetailText>
                {" "}
                - 피그마 UI/UX 디자인 특강
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 앱 개발 동아리와 협업 프로젝트 진행
              </CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 차기 동아리 인수인계 작업
              </CurriculumDetailText>
            </CurriculumBox>
          </CurriculumGroup>
          <CurriculumGroup>
            <CurriculumBox>
              <CurriculumTitle>기타</CurriculumTitle>
              <CurriculumDetailText>
                {" "}
                - 다른 전공 동아리와 협동 수업
              </CurriculumDetailText>
              <CurriculumDetailText> - 인공지능 수학 수업</CurriculumDetailText>
              <CurriculumDetailText> - 알고리즘 수업</CurriculumDetailText>
              <CurriculumDetailText>
                {" "}
                - 자체제작 문제집 과제
              </CurriculumDetailText>
            </CurriculumBox>
          </CurriculumGroup>
        </CurriculumContainer>
      </BigContainer>
    </Container>
  );
};

export default Curriculum;
