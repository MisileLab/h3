import { styled } from "solid-styled-components";
import bg from "../assets/media/background.webp";

const Container = styled.div`
  width: 100%;
  height: 100vh;
  background-image: url(${bg});
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  overflow-x: hidden;
`;

const BlackRect = styled.div`
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.65);
  overflow-x: hidden;
`;

const Title = styled.div`
  color: rgba(255, 255, 255, 0.9);
  font-size: 100px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  @media (max-width: 800px) {
    font-size: 68px;
  }
`;

const SubTitle = styled.div`
  color: #a7a7a7;
  text-align: center;
  font-size: 22px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  letter-spacing: 1.76px;
  margin-bottom: 50px;
  @media (max-width: 800px) {
    font-size: 14px;
  }
`;

const IntroduceContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 7px;
`;

const Home = () => {
  return (
    <Container>
      <BlackRect>
        <IntroduceContainer>
          <Title>P&nbsp;&nbsp;A&nbsp;&nbsp;R&nbsp;&nbsp;A</Title>
          <SubTitle>Project Achievement & Research AI</SubTitle>
        </IntroduceContainer>
      </BlackRect>
    </Container>
  );
};

export default Home;
