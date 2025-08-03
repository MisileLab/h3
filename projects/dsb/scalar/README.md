# Node Explorer

A node-based exploration prototype built with TypeScript, Phaser.js, and Tauri. Explore interconnected nodes on a map, discover resources, battle enemies, and unlock new areas.

## Features

- **Node-based Exploration**: Navigate between interconnected nodes on a map
- **Multiple Node Types**: 
  - Start nodes (beginning points)
  - Exploration nodes (discover new areas)
  - Resource nodes (collect materials)
  - Enemy nodes (battle foes)
  - Treasure nodes (find rewards)
  - Boss nodes (challenging encounters)
  - Exit nodes (progress to next realm)
- **Progression System**: Level up, gain experience, and unlock new areas
- **Resource Management**: Collect and manage various resources
- **Visual Feedback**: Color-coded nodes and smooth animations
- **Cross-platform**: Built with Tauri for desktop deployment

## Technology Stack

- **TypeScript**: Primary programming language
- **Phaser.js**: Game engine for 2D graphics and interactions
- **Tauri**: Desktop application framework
- **pnpm**: Package manager
- **Vite**: Build tool

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- pnpm
- Rust (for Tauri)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pnpm install
   ```

### Development

Run the development server:
```bash
pnpm dev
```

### Building

Build for production:
```bash
pnpm build
```

### Tauri Commands

Run Tauri development:
```bash
pnpm tauri dev
```

Build Tauri application:
```bash
pnpm tauri build
```

## Game Mechanics

### Node States
- **Locked**: Cannot be accessed (requires specific items)
- **Unlocked**: Can be accessed but not explored
- **Explored**: Has been visited
- **Completed**: All content has been finished

### Node Types
- **Start**: Beginning of your journey
- **Exploration**: Discover new areas and gain experience
- **Resource**: Collect valuable materials
- **Enemy**: Battle enemies for rewards
- **Treasure**: Find special rewards (may require keys)
- **Boss**: Challenging encounters with powerful foes
- **Exit**: Progress to the next realm

### Player Progression
- Gain experience by exploring nodes
- Level up to increase health and unlock new abilities
- Collect resources to unlock special areas
- Discover new nodes by completing connected areas

## Project Structure

```
src/
├── main.ts                 # Game initialization
├── game/
│   ├── types/
│   │   └── index.ts       # Type definitions
│   ├── systems/
│   │   └── GameState.ts   # Game state management
│   └── scenes/
│       ├── MainMenuScene.ts      # Main menu
│       ├── ExplorationScene.ts   # Map exploration
│       └── NodeDetailScene.ts    # Node interaction
```

## Controls

- **Mouse**: Click on nodes to navigate
- **Hover**: See node information
- **UI Buttons**: Interact with game features

## Future Enhancements

- Combat system with turn-based battles
- Inventory management
- Save/load functionality
- Sound effects and music
- More complex node networks
- Multiple realms/worlds
- Character customization
- Achievement system

## License

This project is open source and available under the MIT License.
