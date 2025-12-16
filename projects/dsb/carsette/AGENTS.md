# Agent Guidelines for Cassette Tactics

## Build & Development
- `pnpm dev` - Start development server (localhost:3000)
- `pnpm build` - Type check with tsc and build production bundle
- `pnpm preview` - Preview production build
- No test runner configured yet

## Code Style
**Imports**: Use ES modules (`import`), Phaser imported as `import Phaser from 'phaser'`
**Types**: Strict TypeScript - all `strict` flags enabled, no implicit any, no unused locals/parameters
**Formatting**: Use single quotes for strings, explicit return types on methods
**Naming**: PascalCase for classes/scenes, camelCase for variables/methods, SCREAMING_SNAKE_CASE for constants
**Error Handling**: Use proper null checks (`element | null`), optional chaining (`?.`), definite assignment (`!`) only when guaranteed
**Classes**: Use `private`/`public` modifiers, initialize required properties in constructor or with `!` if assigned in lifecycle methods
**Scenes**: Phaser scenes extend `Phaser.Scene`, use `create()` and `update()` lifecycle methods
**Singletons**: Use static getInstance() pattern (see UIManager)
**State Management**: Track game state with typed properties, avoid any type

## Project Structure
- `/src/game/scenes/` - Phaser scene classes
- `/src/game/config.ts` - Game configuration
- `/src/ui/` - DOM-based UI managers
- Entry point: `src/main.ts`
