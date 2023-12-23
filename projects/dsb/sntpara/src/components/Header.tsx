import { Index, createEffect, createSignal } from "solid-js";
import { styled } from "solid-styled-components";

interface ContainerProps {
  isScrolled: boolean;
}

const Container = styled.header<ContainerProps>`
  position: fixed;
  top: ${isScrolled => isScrolled ? "0" : "-200px"};
  transition: top 0.5s ease-in-out;
  width: 100%;
  z-index: 1000;
  display: flex;
  padding: 15px 0px;
  justify-content: center;
  align-items: flex-start;
  background: rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(2px);
  @media (max-width: 1100px) {
    width: 0px;
    height: 0px;
  }
`;

const Button = styled.button`
  display: flex;
  padding: 10px 30px;
  justify-content: center;
  align-items: center;
  gap: 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0);
  border: none;
  transition: background-color 0.2s ease-in-out;

  &:hover {
    background: rgba(170, 170, 170, 0.5);
  }
  @media (max-width: 1100px) {
    width: 0px;
    height: 0px;
  }
`;

const ButtonText = styled.div`
  color: #000;
  font-size: 18px;
  font-style: normal;
  font-weight: 400;
  line-height: normal;
  @media (max-width: 1100px) {
    font-size: 0px;
  }
`;

const buttonTexts = [
  "Home",
  "Introduce",
  "Curriculum",
  "Record",
  "Apply & Enquiry",
];

const Header = () => {
  const [isScrolled, setIsScrolled] = createSignal(false);

  createEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY >= window.innerHeight);
    };

    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const move = (seq: number) => {
    window.scrollTo({
      top: window.innerHeight * seq,
      behavior: "smooth",
    });
  };
  return (
    <Container isScrolled={isScrolled()}>
      <Index each={buttonTexts}>{(a, i)=>
        <Button onClick={()=>move(i)}>
          <ButtonText>{a()}</ButtonText>
        </Button>
      }</Index>
    </Container>
  );
};

export default Header;
