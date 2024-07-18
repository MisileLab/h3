import { h as head, e as pop, p as push } from "../../chunks/index.js";
function _layout($$payload, $$props) {
  push();
  let { children } = $$props;
  head($$payload, ($$payload2) => {
    $$payload2.title = `<title>Animotion</title>`;
  });
  children($$payload);
  $$payload.out += `<!---->`;
  pop();
}
export {
  _layout as default
};
