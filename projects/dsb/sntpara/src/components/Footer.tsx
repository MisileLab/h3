import { styled } from "solid-styled-components";

const Container = styled.div`
  width: 100%;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  height: 220px;
  background: #313131;
  @media (max-width: 500px) {
    height: 110px;
  }
`;

const Text = styled.div`
  text-align: center;
  font-size: 18px;
  color: white;
  line-height: 140%;
  @media (max-width: 800px) {
    font-size: 14px;
  }
`;

const Footer = () => {
  return (
    <Container>
      <Text>
        개발 : 유채호 이정훈
        <br />
        디자인 : 유채호 이정훈
      </Text>
    </Container>
  );
};

export default Footer;
