import React, { useMemo } from "react";

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

/**
 * Color map from white to red
 */
export function redColorMap(value: number, maxValue: number) {
  // Ensure value is between 0 and maxValue
  value = Math.max(0, Math.min(value, maxValue));

  // Normalize value between 0 and 1
  const normalized = value / maxValue;

  // Interpolate between white (255, 255, 255) and red (255, 0, 0)
  const red = 255;
  const green = Math.round(255 * (1 - normalized));
  const blue = Math.round(255 * (1 - normalized));

  // Return the interpolated color as a hex string
  return `rgb(${red}, ${green}, ${blue})`;
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

  return (
    <div className="inline-flex">
      {startIdx > 0 && (
        <span className="relative inline-flex group">
          <span style={{ color: "gray" }}>+{startIdx}</span>
          ...
          {/* Tooltip */}
          <span className="invisible absolute left-full ml-2 top-1/2 -translate-y-1/2 w-max bg-gray-900 text-white text-xs rounded-md px-2 py-1 opacity-0 group-hover:opacity-100 group-hover:visible transition-opacity">
            The number of amino acids hidden in this sequence. (AlphaFoldDB ID:{" "}
            {seq.alphafold_id})
          </span>
        </span>
      )}
      {tokensToShow.map((token, index) => {
        const color = redColorMap(activationsToShow[index], maxValue);
        return (
          <span
            key={`token-${index}`}
            style={{
              backgroundColor: color,
              borderRadius: 2,
              letterSpacing: -1,
            }}
          >
            {token_dict[token]}
          </span>
        );
      })}
    </div>
  );
};

export default SeqViewer;
