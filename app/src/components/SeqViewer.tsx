import React, { useMemo } from "react";
import MolstarViewer from "./MolstarViewer";

export interface SeqFormat {
  dimension: number;
  feature_name?: string;
  examples: SingleSeq[];
}

interface SingleSeq {
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

const token_dict = {
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

/**
 * This component takes in a SingleSeq and renders the sequence with the corresponding colors
 */
const SeqViewer: React.FC<SeqViewerProps> = ({ seq }) => {
  console.log(seq);
  const maxValue = useMemo(() => {
    return Math.max(...seq.tokens_acts_list);
  }, [seq.tokens_acts_list]);

  return (
    <div>
      <MolstarViewer alphafold_id={seq.alphafold_id} activation_list={seq.tokens_acts_list} />
      {seq.alphafold_id}{" "}
      {seq.tokens_list.map((token, index) => {
        const color = redColorMap(seq.tokens_acts_list[index], maxValue);
        return (
          <span
            style={{
              backgroundColor: color,
              borderRadius: 2,
              letterSpacing: -1,
            }}
          >
            {token_dict[seq.tokens_list[index]]}
          </span>
        );
      })}
    </div>
  );
};

export default SeqViewer;
