import React, { useMemo } from "react";

interface SeqFormat {
  quantile_name: string;
  examples: SingleSeq[];
}

interface SingleSeq {
  tokens_acts_list: Array<number>;
  tokens_id_list: Array<number>;
}

interface SeqViewerProps {
  seq: SingleSeq;
}

/**
 * Color map from white to red
 */
function redColorMap(value: number, maxValue: number) {
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
      {seq.tokens_id_list.map((token_id, index) => {
        const color = redColorMap(seq.tokens_acts_list[index], maxValue);
        return <span style={{ backgroundColor: color, borderRadius: 2 }}>{token_id}</span>;
      })}
    </div>
  );
};

export default SeqViewer;
