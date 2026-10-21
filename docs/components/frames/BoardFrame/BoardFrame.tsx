import type * as React from "react";
import FrameShell from "@/components/frames/FrameShell/FrameShell";
import type { LegendEntry } from "@/components/frames/FrameShell/FrameShell";
import type { BoardData, BoardType } from "@/lib/frames/board";
import { BOARD_KIND } from "@/lib/frames/kinds";
import BoardPlot from "./_BoardPlot";
import { layoutBoard } from "./_layout";

interface BoardFrameProps {
  id: string;
  board?: BoardData;
  type: string;
  title: string;
  description: string;
  kicker?: string;
  subtitle?: string;
  note?: string;
  caption?: string;
}

const BoardFrame: React.FC<BoardFrameProps> = ({ id, board, type, title, description, kicker, subtitle, note, caption }) => {
  const problems = !board
    ? ["unknown or invalid data"]
    : !BOARD_KIND.types.includes(type)
      ? [`unknown type "${type}"`]
      : BOARD_KIND.typeProblems(type, board);
  if (!board || problems.length) {
    console.warn(`[BoardFrame] Can't draw "${id}" as "${type}": ${problems.join("; ")}. Run yarn check:content for details.`);
    return null;
  }
  // A venn's members name their category; each named color gets a legend entry.
  const legend: LegendEntry[] = [];
  for (const m of board.main.members)
    if (m.label && m.color && !legend.some(e => e.label === m.label)) legend.push({ id: m.label, label: m.label, color: m.color });
  return (
    <FrameShell
      data={{ "data-board": id, "data-board-type": type }}
      legend={legend}
      title={title}
      description={description}
      kicker={kicker}
      subtitle={subtitle}
      caption={caption}
      note={note}
    >
      <BoardPlot scene={layoutBoard(board, type as BoardType)} id={id} />
    </FrameShell>
  );
};

export const createBoardFrame = (data: Record<string, BoardData>) =>
  function ArticleBoardFrame(props: Omit<BoardFrameProps, "board">) {
    return <BoardFrame {...props} board={Object.hasOwn(data, props.id) ? data[props.id] : undefined} />;
  };

export default BoardFrame;
