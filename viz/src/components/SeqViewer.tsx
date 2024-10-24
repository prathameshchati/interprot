import React, { useMemo } from "react";
import { redColorMap } from "../utils";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

export interface SeqFormat {
  dimension: number;
  feature_name?: string;
  examples: SingleSeq[];
}

export interface SingleSeq {
  alphafold_id: string;
  tokens_acts_list: Array<number>;
  tokens_list: Array<number>;
}

interface SeqViewerProps {
  seq: SingleSeq;
}

const token_dict: { [key: number]: string } = {
  4: "L",
  5: "A",
  6: "G",
  7: "V",
  8: "S",
  9: "E",
  10: "R",
  11: "T",
  12: "I",
  13: "D",
  14: "P",
  15: "K",
  16: "Q",
  17: "N",
  18: "F",
  19: "Y",
  20: "M",
  21: "H",
  22: "W",
  23: "C",
  24: "X",
  25: "B",
  26: "U",
  27: "Z",
  28: "O",
  29: ".",
  30: "-",
};

function getFirstNonZeroIndex(arr: Array<number>) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] !== 0) {
      return i;
    }
  }
  return 0;
}

/**
 * This component takes in a SingleSeq and renders the sequence with the corresponding colors
 */
const SeqViewer: React.FC<SeqViewerProps> = ({ seq }) => {
  const maxValue = useMemo(() => {
    return Math.max(...seq.tokens_acts_list);
  }, [seq.tokens_acts_list]);
  const firstNonZero = getFirstNonZeroIndex(seq.tokens_acts_list);

  const startIdx = Math.max(firstNonZero - 30, 0);

  const tokensToShow = seq.tokens_list.slice(startIdx);
  const activationsToShow = seq.tokens_acts_list.slice(startIdx);

  const copySequenceToClipboard = () => {
    navigator.clipboard.writeText(seq.tokens_list.map((token) => token_dict[token]).join(""));
  };

  return (
    <div className="inline-flex" style={{ cursor: "pointer" }}>
      {startIdx > 0 && (
        <TooltipProvider delayDuration={100}>
          <Tooltip>
            <TooltipTrigger>
              <span style={{ color: "gray" }}>+{startIdx}</span>
              ...
            </TooltipTrigger>
            <TooltipContent>
              The number of amino acids hidden in this sequence. (AlphaFoldDB ID: {seq.alphafold_id}
              )
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
      {tokensToShow.map((token, index) => {
        const color = redColorMap(activationsToShow[index], maxValue);
        const tokenChar = token_dict[token];
        return (
          <TooltipProvider key={`token-${index}`} delayDuration={100}>
            <Tooltip>
              <TooltipTrigger>
                <span
                  style={{
                    backgroundColor: color,
                    borderRadius: 2,
                    letterSpacing: -1,
                  }}
                >
                  {tokenChar}
                </span>
              </TooltipTrigger>
              <TooltipContent onClick={copySequenceToClipboard} style={{ cursor: "pointer" }}>
                Copy
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        );
      })}
    </div>
  );
};

export default SeqViewer;
