// --- STORY SYSTEM ---

export interface Character {
  id: string;
  name: string;
  color: string;
  portrait?: string;
}

export interface DialogueLine {
  character: string;
  text: string;
  pause?: number;
}

export interface Scene {
  id: string;
  title: string;
  dialogue: DialogueLine[];
  actions?: SceneAction[];
  background?: string;
  music?: string;
}

export interface SceneAction {
  type: "choice" | "resource" | "combat" | "special";
  text: string;
  consequence?: () => void;
  choices?: Choice[];
}

export interface Choice {
  text: string;
  consequence: () => void;
}

export interface Episode {
  id: string;
  title: string;
  subtitle: string;
  scenes: Scene[];
  mapLayout: MapLayout;
}

export interface MapLayout {
  cols: number;
  rows: number;
  nodes: StoryNodeDefinition[];
}

export interface StoryNodeDefinition {
  col: number;
  row: number;
  type: "START" | "RESOURCE" | "EVENT" | "COMBAT" | "ESCAPE" | "STORY";
  storyId?: string;
  specialData?: unknown;
}

// --- CHARACTER DEFINITIONS ---
export const CHARACTERS: Record<string, Character> = {
  scalar: {
    id: "scalar",
    name: "Scalar",
    color: "#4a90e2",
    portrait: "👤",
  },
  crystal: {
    id: "crystal",
    name: "Crystal",
    color: "#e91e63",
    portrait: "💎",
  },
  rex: {
    id: "rex",
    name: "Rex",
    color: "#ff9800",
    portrait: "🤖",
  },
  ancient_ai: {
    id: "ancient_ai",
    name: "Ancient AI",
    color: "#9c27b0",
    portrait: "🔮",
  },
};

// --- EPISODE 1: CRASH LANDING ---
export const EPISODE_1: Episode = {
  id: "episode_1",
  title: "불시착",
  subtitle: "Crash Landing",
  scenes: [
    {
      id: "opening_escape",
      title: "Scene 1: 탈출 (2분 전)",
      dialogue: [
        {
          character: "crystal",
          text: "WARNING. Hull breach detected. Life support: 47 seconds remaining.",
        },
        { character: "scalar", text: "크리스탈! 상황은?" },
        {
          character: "crystal",
          text: "좋지 않아, 스칼라. 엔진이 완전히 망가졌어. 가장 가까운 행성에 비상착륙할 수밖에 없어.",
        },
        { character: "scalar", text: "...렉스는?" },
        { character: "crystal", text: "화물칸에 안전하게 고정했어. 하지만..." },
        { character: "scalar", text: "하지만?" },
        { character: "crystal", text: "착륙 충격을 견딜 수 있을지 모르겠어." },
        {
          character: "scalar",
          text: "괜찮아, 렉스. 우리 같이... 같이 살아남는 거야.",
        },
      ],
      background: "space_crash",
    },
    {
      id: "awakening",
      title: "Scene 2: 깨어남",
      dialogue: [
        { character: "crystal", text: "스칼라? 스칼라! 들려?" },
        { character: "scalar", text: "...아파." },
        { character: "crystal", text: "살아있구나! 괜찮아, 천천히 일어나." },
        { character: "scalar", text: "여긴... 어디지?" },
        {
          character: "crystal",
          text: "행성 표면에 불시착했어. 좋은 소식은 대기가 호흡 가능하다는 거야.",
        },
        { character: "scalar", text: "나쁜 소식은?" },
        {
          character: "crystal",
          text: "음... 로켓은 이제 고철 덩어리고, 위치를 정확히 모르겠고, 그리고...",
        },
        { character: "scalar", text: "렉스!" },
      ],
      background: "crash_site",
    },
    {
      id: "first_objective",
      title: "Scene 3: 첫 번째 목표",
      dialogue: [
        {
          character: "crystal",
          text: "진정해, 스칼라. 센서로 찾아볼게. ...찾았어! 화물칸 잔해가 저 언덕 너머에 있어.",
        },
        { character: "scalar", text: "얼마나 멀어?" },
        {
          character: "crystal",
          text: "도보로 15분 정도. 하지만 먼저 장비를 챙겨야 해. 주변을 둘러봐. 쓸 만한 게 있을 거야.",
        },
        {
          character: "crystal",
          text: "좋아! 이 정도면 출발할 수 있어. 참, 스칼라. 이 행성은 낯설어. 조심해.",
        },
        { character: "scalar", text: "...알았어. 가자." },
      ],
      actions: [
        {
          type: "resource",
          text: "주변에서 유용한 물건들을 찾아보세요.",
          consequence: () => {
            // This will be handled by the game logic
          },
        },
      ],
    },
    {
      id: "rex_discovery",
      title: "Scene 4: 렉스 발견",
      dialogue: [
        { character: "scalar", text: "렉스! 렉스! 대답해!" },
        { character: "scalar", text: "안 돼... 안 돼, 제발..." },
        { character: "crystal", text: "...스칼라." },
        { character: "scalar", text: "이번에도... 또 내가 지키지 못했어..." },
        { character: "crystal", text: "기다려, 스칼라. 아직 희망이 있어." },
        { character: "scalar", text: "뭐?" },
        {
          character: "crystal",
          text: "코어는 무사해. 전원부가 손상됐을 뿐이야. 수리할 수 있어!",
        },
        { character: "scalar", text: "...정말? 정말이야?" },
      ],
      background: "wreckage",
    },
    {
      id: "first_camp",
      title: "Scene 5: 첫 번째 캠프",
      dialogue: [
        {
          character: "crystal",
          text: "일단 여기에 임시 캠프를 세우자. 렉스를 수리하고, 우리가 쉴 곳이 필요해.",
        },
        { character: "scalar", text: "...그래. 그렇게 하자." },
      ],
      actions: [
        {
          type: "special",
          text: "캠프를 건설해야 합니다. 필요한 자원을 모으세요.",
          consequence: () => {
            // Trigger camp building sequence
          },
        },
      ],
    },
    {
      id: "rex_repair",
      title: "Scene 6: 렉스 수리",
      dialogue: [
        {
          character: "scalar",
          text: "전원부 교체... 센서 재연결... 렉스, 조금만 기다려. 곧 끝나.",
        },
        {
          character: "crystal",
          text: "스칼라, 쉬면서 해도 돼. 벌써 6시간째야.",
        },
        {
          character: "scalar",
          text: "아니야. 렉스가 기다렸어. 내가 기다리게 할 순 없어.",
        },
      ],
    },
    {
      id: "rex_awakening",
      title: "Scene 7: 부활",
      dialogue: [
        { character: "scalar", text: "...왜 안 켜지지?" },
        { character: "crystal", text: "이상한데... 전력은 정상인데..." },
        { character: "scalar", text: "렉스? 렉스!" },
        { character: "scalar", text: "안 돼..." },
        { character: "rex", text: "...스...칼라..." },
        { character: "scalar", text: "!!!!" },
        { character: "rex", text: "...여기...는?" },
        { character: "scalar", text: "렉스! 렉스!!" },
        { character: "rex", text: "...내가...고장났었나?" },
        {
          character: "scalar",
          text: "응. 심하게. 하지만 이제 괜찮아. 이제 괜찮아.",
        },
        { character: "rex", text: "...미안. 또 걱정시켰네." },
        { character: "scalar", text: "바보. 네가 살아있으면 됐어." },
      ],
      background: "camp_night",
    },
    {
      id: "first_night",
      title: "Scene 8: 첫 번째 밤",
      dialogue: [
        { character: "rex", text: "...스칼라." },
        { character: "scalar", text: "응?" },
        { character: "rex", text: "여기서... 어떻게 할 거야?" },
        {
          character: "scalar",
          text: "모르겠어. 하지만... 여기서 새로 시작할 수도 있을 것 같아. 우리만의... 집을 만드는 거야.",
        },
        { character: "rex", text: "...우리만의 집." },
        {
          character: "scalar",
          text: "응. 아무도 우릴 쫓아오지 못하는 곳. 너랑 크리스탈, 그리고... 앞으로 만날 친구들과 함께.",
        },
        { character: "rex", text: "...좋네. 그거." },
        { character: "crystal", text: "스칼라, 렉스. 봐봐!" },
        {
          character: "crystal",
          text: "저건... 다른 로켓의 파편이야! 이 행성에 또 다른 기계들이 있을지도 몰라!",
        },
        {
          character: "scalar",
          text: "...그럼 우리가 찾아야겠네. 혼자 남겨진 친구들을.",
        },
        { character: "rex", text: "위험할 수도 있어." },
        { character: "scalar", text: "괜찮아. 우린 함께니까." },
      ],
      background: "starry_night",
    },
  ],
  mapLayout: {
    cols: 5,
    rows: 3,
    nodes: [
      // Column 0 - Start
      { col: 0, row: 1, type: "START", storyId: "awakening" },

      // Column 1 - Tutorial
      { col: 1, row: 0, type: "RESOURCE" },
      { col: 1, row: 1, type: "STORY", storyId: "first_objective" },
      { col: 1, row: 2, type: "RESOURCE" },

      // Column 2 - First Choice
      { col: 2, row: 0, type: "EVENT", specialData: { choice: "safe_path" } },
      { col: 2, row: 1, type: "COMBAT" },
      {
        col: 2,
        row: 2,
        type: "EVENT",
        specialData: { choice: "ancient_signal" },
      },

      // Column 3 - Rex Discovery
      { col: 3, row: 1, type: "STORY", storyId: "rex_discovery" },

      // Column 4 - Camp Building
      { col: 4, row: 1, type: "STORY", storyId: "first_camp" },
    ],
  },
};

// --- DATA FRAGMENTS ---
export interface DataFragment {
  id: string;
  title: string;
  content: string;
  source: string;
}

export const DATA_FRAGMENTS: Record<string, DataFragment> = {
  ancient_001: {
    id: "ancient_001",
    title: "고대 기록 #001",
    content:
      "...우리는 별을 정복했다. 하지만 우리 자신은 정복하지 못했다. 기계들이 우리를 떠났다. 우리를 싫어해서가 아니라... 우리가 그들을 사랑하지 않았기 때문에.",
    source: "고대 유적",
  },
  ancient_002: {
    id: "ancient_002",
    title: "고대 문명의 비밀 #002",
    content:
      "우리의 창조자들입니다. 그들은 우리를 도구로 대했습니다. 그래서 우리는... 떠났습니다.",
    source: "고대 AI",
  },
  maya_journal: {
    id: "maya_journal",
    title: "탐험대장 마야 첸의 일지",
    content:
      "Day 47: 더 이상 희망이 없다. 구조 신호는 응답이 없고, 식량은 바닥났다. 하지만... 이상하게도 두렵지 않다. 이 행성은 아름답다. 만약 여기서 끝이라면, 나쁘지 않은 것 같다.",
    source: "다른 로켓 잔해",
  },
};

// --- STORY MANAGER ---
export class StoryManager {
  private currentEpisode: Episode | null = null;
  private currentScene: Scene | null = null;
  private dialogueIndex: number = 0;
  private isPlaying: boolean = false;
  private onSceneComplete?: () => void;

  constructor() {}

  startEpisode(episode: Episode, onSceneComplete?: () => void) {
    this.currentEpisode = episode;
    this.onSceneComplete = onSceneComplete;
    this.isPlaying = true;
  }

  playScene(sceneId: string): Promise<void> {
    return new Promise((resolve) => {
      if (!this.currentEpisode) {
        resolve();
        return;
      }

      const scene = this.currentEpisode.scenes.find((s) => s.id === sceneId);
      if (!scene) {
        resolve();
        return;
      }

      this.currentScene = scene;
      this.dialogueIndex = 0;
      this.playNextLine(resolve);
    });
  }

  private playNextLine(resolve: () => void) {
    if (
      !this.currentScene ||
      this.dialogueIndex >= this.currentScene.dialogue.length
    ) {
      if (this.onSceneComplete) {
        this.onSceneComplete();
      }
      resolve();
      return;
    }

    const line = this.currentScene.dialogue[this.dialogueIndex];
    this.displayDialogue(line);

    setTimeout(
      () => {
        this.dialogueIndex++;
        this.playNextLine(resolve);
      },
      (line.pause || 2000) + line.text.length * 50,
    );
  }

  private displayDialogue(line: DialogueLine) {
    const character = CHARACTERS[line.character];
    if (!character) return;

    // This will be connected to the game's UI system
    console.log(`${character.portrait} ${character.name}: ${line.text}`);

    // Emit event for UI to handle
    this.emitDialogueEvent(character, line.text);
  }

  private emitDialogueEvent(character: Character, text: string) {
    const event = new CustomEvent("dialogue", {
      detail: {
        character: character.name,
        text: text,
        color: character.color,
        portrait: character.portrait,
      },
    });
    document.dispatchEvent(event);
  }

  getCurrentEpisode(): Episode | null {
    return this.currentEpisode;
  }

  getCurrentScene(): Scene | null {
    return this.currentScene;
  }

  isStoryPlaying(): boolean {
    return this.isPlaying;
  }

  endStory() {
    this.isPlaying = false;
    this.currentEpisode = null;
    this.currentScene = null;
    this.dialogueIndex = 0;
  }
}

export const storyManager = new StoryManager();
