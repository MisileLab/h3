import { p as push, k as attr, e as pop, l as spread_attributes, m as stringify, o as bind_props, f as escape_html } from "../../chunks/index.js";
function Action($$payload, $$props) {
  push();
  let { order, ...props } = $$props;
  $$payload.out += `<div class="fragment hidden"${attr("data-fragment-index", order)}></div>`;
  pop();
}
function Code($$payload, $$props) {
  push();
  let {
    code,
    lang,
    theme = "poimandres",
    options = {},
    autoIndent = true,
    ...props
  } = $$props;
  let container;
  const is = {
    htmlEl: (el) => el instanceof HTMLElement,
    token: (el) => el.className.includes("shiki-magic-move-item"),
    newLine: (el) => el.tagName === "BR"
  };
  async function render(code2) {
    return;
  }
  function update(code2) {
    return render(code2[0]);
  }
  function createRange(start, end) {
    return Array.from({ length: end - start + 1 }, (_, index) => start + index);
  }
  function getLines(string) {
    let range = string[0];
    if (range === "*") {
      return [];
    }
    return range.split(",").flatMap((number) => {
      if (number.includes("-")) {
        const [start, end] = number.split("-");
        return createRange(+start, +end);
      } else {
        return +number;
      }
    });
  }
  function transition(el, selected) {
    const { promise, resolve } = Promise.withResolvers();
    const selectToDeselect = !selected && el.classList.contains("selected");
    const deselectToSelect = selected && el.classList.contains("deselected");
    const nothingToDeselect = !selected && !el.classList.contains("deselected");
    const willTransition = selectToDeselect || deselectToSelect || nothingToDeselect;
    if (willTransition) {
      el.ontransitionend = resolve;
    } else {
      resolve("finished");
    }
    el.classList.toggle("selected", selected);
    el.classList.toggle("deselected", !selected);
    return promise;
  }
  function selectLines(string) {
    const lines = getLines(string);
    const tokens = container.children;
    const promises = [];
    let currentLine = 1;
    for (const token of tokens) {
      if (!is.htmlEl(token)) {
        return;
      }
      if (is.token(token)) {
        let selected = false;
        lines.length === 0 ? selected = true : selected = lines.includes(currentLine);
        promises.push(transition(token, selected));
      }
      if (is.newLine(token)) {
        currentLine++;
      }
    }
    return Promise.all(promises);
  }
  function selectToken(string) {
    const selection = string[0].split(" ");
    const isLineNumber = !isNaN(+selection[0]);
    const line = isLineNumber ? +selection[0] : 0;
    const tokens = container.children;
    const promises = [];
    let currentLine = 1;
    for (const token of tokens) {
      if (!is.htmlEl(token)) return;
      if (is.token(token)) {
        let selected = false;
        if (isLineNumber && line === currentLine) {
          selected = selection.includes(token.textContent);
        }
        if (!isLineNumber) {
          selected = selection.includes(token.textContent);
        }
        promises.push(transition(token, selected));
      }
      if (is.newLine(token)) {
        currentLine++;
      }
    }
    return Promise.all(promises);
  }
  $$payload.out += `<pre${spread_attributes({
    ...props,
    class: `shiki-magic-move-container ${stringify(props.class)}`
  })}></pre>`;
  bind_props($$props, { update, selectLines, selectToken });
  pop();
}
function Presentation($$payload, $$props) {
  push();
  let { children, options, ...props } = $$props;
  $$payload.out += `<div class="reveal"><div${attr("class", `slides ${stringify(props.class)}`)}>`;
  if (children) {
    $$payload.out += "<!--[-->";
    children($$payload);
    $$payload.out += `<!---->`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--></div></div>`;
  pop();
}
function Slide($$payload, $$props) {
  push();
  let { children, ...props } = $$props;
  $$payload.out += `<section${attr("data-auto-animate", props.animate)}${attr("data-auto-animate-easing", props.animateEasing)}${attr("data-auto-animate-unmatched", props.animateUnmatched)}${attr("data-auto-animate-id", props.animateId)}${attr("data-auto-animate-restart", props.animateRestart)}${attr("data-background-color", props.background)}${attr("data-background-gradient", props.gradient)}${attr("data-background-image", props.image)}${attr("data-background-video", props.video)}${attr("data-background-iframe", props.iframe)}${attr("data-background-interactive", props.interactive)}${attr("data-transition", props.transition)}${attr("class", props.class)}>`;
  if (children) {
    $$payload.out += "<!--[-->";
    children($$payload);
    $$payload.out += `<!---->`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--></section>`;
  pop();
}
function _page($$payload) {
  let code;
  $$payload.out += `<link href="https://cdn.jsdelivr.net/gh/sun-typeface/SUIT@2/fonts/variable/woff2/SUIT-Variable.css" rel="stylesheet" class="svelte-1mhdbvk"> `;
  Presentation($$payload, {
    options: {
      history: true,
      transition: "fade",
      controls: true,
      progress: true
    },
    children: ($$payload2, $$slotProps) => {
      Slide($$payload2, {
        children: ($$payload3, $$slotProps2) => {
          $$payload3.out += `<section class="svelte-1mhdbvk"></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-screen h-full justify-center items-center gap-10 svelte-1mhdbvk"><img src="ecoProgressBefore.png" alt="" width="200%" class="svelte-1mhdbvk"> <span class="w-fit grow whitespace-nowrap svelte-1mhdbvk">${escape_html(`->`)}</span> <img src="ecoProgressAfter.png" alt="" width="200%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">사람이 위험한 행동을 하면 오염도가 늘어납니다.</p></section> <section class="flex-col justify-center w-full h-screen svelte-1mhdbvk"><div class="flex w-screen h-full justify-center items-center svelte-1mhdbvk"><img src="ecoProgressFull.png" alt="" width="50%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">오염도가 일정 수준에 다다르면 패배합니다.</p></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-full h-full justify-center items-center gap-10 svelte-1mhdbvk"><img src="thinking.png" alt="" width="30%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">행동을 실행하기 전에 고민을 하고,</p></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-full h-full justify-center items-center gap-10 svelte-1mhdbvk"><img src="acting.png" alt="" width="30%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">그 후에 위험한 행동을 합니다.</p></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-screen h-full justify-center items-center gap-10 svelte-1mhdbvk"><img src="item.png" alt="" width="30%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">아이템을 사용해서 오염도를 낮출 수 있습니다.</p></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-screen h-full justify-center items-center gap-10 svelte-1mhdbvk"><img src="itemUseBefore.png" alt="" width="200%" class="svelte-1mhdbvk"> <span class="w-fit grow whitespace-nowrap svelte-1mhdbvk">${escape_html(`->`)}</span> <img src="itemUseAfter.png" alt="" width="200%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">아이템은 일정 코스트를 필요로 합니다.</p></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-screen h-full justify-center items-center gap-10 svelte-1mhdbvk"><img src="kickButton.png" alt="" width="50%" class="svelte-1mhdbvk"></div> <p class="text-7xl svelte-1mhdbvk">또는 문제의 근원을 제거할 수도 있습니다.</p></section> <section class="flex-col justify-center w-full h-full svelte-1mhdbvk"><div class="flex w-screen h-full justify-center items-center gap-10 svelte-1mhdbvk"><p class="text-7xl svelte-1mhdbvk">시간이 흐를 수록 사람의 수와 행동 빈도가 증가합니다.</p></div></section>`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Slide($$payload2, {
        transition: "none",
        children: ($$payload3, $$slotProps2) => {
          $$payload3.out += `<section class="flex-col h-screen items-center svelte-1mhdbvk"><div class="svelte-1mhdbvk">`;
          Code($$payload3, {
            code: `...
          readonly Func<float, float, float> _getIntensity = (x, intensity) => intensity * (Camera.main.orthographicSize / 25) * Time.deltaTime * x;
          private readonly Func<float, float, bool> _accurateFloatCompare = (f, f2) => Math.Abs(f - f2) <= Epsilon;
        
          ...
        
          void MoveCamera(Func<float, float> callbackx, Func<float, float> callbacky) {
              if (Input.GetMouseButton(1) && Input.mousePosition != _pastMousePos) {
                  Vector3 velocity = _pastMousePos-Input.mousePosition;
                  if (!_accurateFloatCompare(Input.mousePosition.x, _pastMousePos.x)) {
                    transform.Translate(callbackx(velocity.x), 0, 0);
                  }
                  if (!_accurateFloatCompare(Input.mousePosition.y, _pastMousePos.y)) {
                    transform.Translate(0, callbacky(velocity.y), 0);
                  }
              }
          }
        }`,
            lang: "csharp",
            theme: "catppuccin-mocha",
            options: { lineNumbers: true }
          });
          $$payload3.out += `<!----> `;
          Action($$payload3, { do: () => code.selectLines`7-18` });
          $$payload3.out += `<!----> `;
          Action($$payload3, {
            do: () => code.update`if (!isIgnoringMove) {
        MoveCamera(
            x => _getIntensity(x, camIntensity) * _movePerSecond.x,
            y => _getIntensity(y, camIntensity) * _movePerSecond.y
        );
        ...
        }`
          });
          $$payload3.out += `<!----> `;
          Action($$payload3, { do: () => code.selectToken`MoveCamera` });
          $$payload3.out += `<!----> `;
          Action($$payload3, { do: () => code.selectToken`_getIntensity` });
          $$payload3.out += `<!----> `;
          Action($$payload3, {
            do: () => code.update`...
[System.Serializable]
public class Node
{
    public Node(bool _isWall, int _x, int _y) { isWall = _isWall; x = _x; y = _y; }

    public bool isWall;
    public Node ParentNode;

    // G : 시작으로부터 이동했던 거리, H : |가로|+|세로| 장애물 무시하여 목표까지의 거리, F : G + H
    public int x, y, G, H;
    public int F { get { return G + H; } }
}`
          });
          $$payload3.out += `<!----> `;
          Action($$payload3, {
            do: () => code.update`...
public List<Node> FinalNodeList;
Node[,] NodeArray;
int sizeX, sizeY;
Node StartNode, TargetNode, CurNode;
List<Node> OpenList, ClosedList;
public Vector3 dest;
public int i = 0;`
          });
          $$payload3.out += `<!----> `;
          Action($$payload3, {
            do: () => code.update`...
public void PathFinding()
{
    isPathFinding = true;

    // NodeArray의 크기 정해주고, isWall, x, y 대입
    sizeX = topRight.x - bottomLeft.x + 1;
    sizeY = topRight.y - bottomLeft.y + 1;
    NodeArray = new Node[sizeX, sizeY];

    for (int i = 0; i < sizeX; i++)
    {
        for (int j = 0; j < sizeY; j++)
        {
            bool isWall = false;
            foreach (Collider2D col in Physics2D.OverlapCircleAll(new Vector2(i + bottomLeft.x, j + bottomLeft.y), 0.4f)) {
                if (col.gameObject.layer == LayerMask.NameToLayer("Wall")) isWall = true;
            }

            NodeArray[i, j] = new Node(isWall, i + bottomLeft.x, j + bottomLeft.y);
        }
    }`
          });
          $$payload3.out += `<!----> `;
          Action($$payload3, {
            do: () => code.update`// 시작과 끝 노드, 열린리스트와 닫힌리스트, 마지막리스트 초기화
    StartNode = NodeArray[startPos.x - bottomLeft.x, startPos.y - bottomLeft.y];
    TargetNode = NodeArray[targetPos.x - bottomLeft.x, targetPos.y - bottomLeft.y];

    OpenList = new List<Node>() { StartNode };
    ClosedList = new List<Node>();
    FinalNodeList = new List<Node>();


    while (OpenList.Count > 0 && isPathFinding)
    {
        // 열린리스트 중 가장 F가 작고 F가 같다면 H가 작은 걸 현재노드로 하고 열린리스트에서 닫힌리스트로 옮기기
        CurNode = OpenList[0];
        for (int i = 0; i < OpenList.Count; i++)
            if (OpenList[i].F < CurNode.F || (OpenList[i].F == CurNode.F && OpenList[i].H < CurNode.H)) {
                CurNode = OpenList[i];
            }
        OpenList.Remove(CurNode);
        ClosedList.Add(CurNode);

        // 마지막
        if (CurNode == TargetNode)
        {
            Node TargetCurNode = TargetNode;
            while (TargetCurNode != StartNode)
            {
                FinalNodeList.Add(TargetCurNode);
                TargetCurNode = TargetCurNode.ParentNode;
            }
            FinalNodeList.Add(StartNode);
            FinalNodeList.Reverse();

            //for (int i = 0; i < FinalNodeList.Count; i++) print(i + "번째는 " + FinalNodeList[i].x + ", " + FinalNodeList[i].y);
        }`
          });
          $$payload3.out += `<!----> `;
          Action($$payload3, {
            do: () => code.update`// ↗↖↙↘
        if (allowDiagonal)
        {
            OpenListAdd(CurNode.x + 1, CurNode.y + 1);
            OpenListAdd(CurNode.x - 1, CurNode.y + 1);
            OpenListAdd(CurNode.x - 1, CurNode.y - 1);
            OpenListAdd(CurNode.x + 1, CurNode.y - 1);
        }

        // ↑ → ↓ ←
        OpenListAdd(CurNode.x, CurNode.y + 1);
        OpenListAdd(CurNode.x + 1, CurNode.y);
        OpenListAdd(CurNode.x, CurNode.y - 1);
        OpenListAdd(CurNode.x - 1, CurNode.y);
    }
    i = 0;
}
`
          });
          $$payload3.out += `<!----></div></section>`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!---->`;
}
export {
  _page as default
};
