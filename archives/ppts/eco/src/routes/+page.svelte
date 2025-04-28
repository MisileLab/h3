<script lang="ts">
	import { Presentation, Slide, Code, Action } from '@animotion/core'

	let code: Code
</script>

<link href="https://cdn.jsdelivr.net/gh/sun-typeface/SUIT@2/fonts/variable/woff2/SUIT-Variable.css" rel="stylesheet">

<style>
  * {font-family: 'SUIT Variable', sans-serif;}
</style>

<Presentation options={{ history: true, transition: 'fade', controls: true, progress: true }}>
  <Slide>
    <section></section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="ecoProgressBefore.png" alt="" width="200%" />
        <span class="w-fit grow whitespace-nowrap">{`->`}</span>
        <img src="ecoProgressAfter.png" alt="" width="200%" />
      </div>
      <p class="text-7xl">사람이 위험한 행동을 하면 오염도가 늘어납니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-screen">
      <div class="flex w-screen h-full justify-center items-center">
        <img src="ecoProgressFull.png" alt="" width="50%" />
      </div>
      <p class="text-7xl">오염도가 일정 수준에 다다르면 패배합니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-full h-full justify-center items-center gap-10">
        <img src="thinking.png" alt="" width="30%" />
      </div>
      <p class="text-7xl">행동을 실행하기 전에 고민을 하고,</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-full h-full justify-center items-center gap-10">
        <img src="acting.png" alt="" width="30%" />
      </div>
      <p class="text-7xl">그 후에 위험한 행동을 합니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="item.png" alt="" width="30%" />
      </div>
      <p class="text-7xl">아이템을 사용해서 오염도를 낮출 수 있습니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="itemUseBefore.png" alt="" width="200%" />
        <span class="w-fit grow whitespace-nowrap">{`->`}</span>
        <img src="itemUseAfter.png" alt="" width="200%" />
      </div>
      <p class="text-7xl">아이템은 일정 코스트를 필요로 합니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="kickButton.png" alt="" width="50%" />
      </div>
      <p class="text-7xl">또는 문제의 근원을 제거할 수도 있습니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <p class="text-7xl">시간이 흐를 수록 사람의 수와 행동 빈도가 증가합니다.</p>
      </div>
    </section>
  </Slide>
  <Slide transition="none">
    <section class="flex-col h-screen items-center">
      <div>
        <Code bind:this={code} code={`...
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
        }`} lang="csharp" theme="catppuccin-mocha" options={{lineNumbers: true}}/>
            <Action do={()=>code.selectLines`7-18`} />
            <Action do={()=>code.update`if (!isIgnoringMove) {
        MoveCamera(
            x => _getIntensity(x, camIntensity) * _movePerSecond.x,
            y => _getIntensity(y, camIntensity) * _movePerSecond.y
        );
        ...
        }`} />
        <Action do={()=>code.selectToken`MoveCamera`} />
        <Action do={()=>code.selectToken`_getIntensity`} />
        <Action do={()=>code.update`...
[System.Serializable]
public class Node
{
    public Node(bool _isWall, int _x, int _y) { isWall = _isWall; x = _x; y = _y; }

    public bool isWall;
    public Node ParentNode;

    // G : 시작으로부터 이동했던 거리, H : |가로|+|세로| 장애물 무시하여 목표까지의 거리, F : G + H
    public int x, y, G, H;
    public int F { get { return G + H; } }
}`} />
      <Action do={()=>code.update`...
public List<Node> FinalNodeList;
Node[,] NodeArray;
int sizeX, sizeY;
Node StartNode, TargetNode, CurNode;
List<Node> OpenList, ClosedList;
public Vector3 dest;
public int i = 0;`} />
      <Action do={()=>code.update`...
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
    }`} />

      <Action do={()=>code.update`// 시작과 끝 노드, 열린리스트와 닫힌리스트, 마지막리스트 초기화
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
        }`} />


        <Action do={()=>code.update`// ↗↖↙↘
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
`} />
      </div>
    </section>
  </Slide>
</Presentation>
