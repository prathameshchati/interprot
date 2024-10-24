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

// PDBeMolstarPlugin doesn't allow color in rgb format
export function rgbToHex(rgb: string): string {
  const [r, g, b] = rgb.match(/\d+/g)!.map(Number);
  const hex = ((r << 16) | (g << 8) | b).toString(16).padStart(6, "0");
  return `#${hex}`;
}

// Generates the "color" data given the activation list
export function residueColor(activation_list: Array<number>) {
  const max_activation = Math.max(...activation_list);
  return activation_list.map((activation, i) => ({
    struct_asym_id: "A",
    residue_number: i + 1,
    color: rgbToHex(redColorMap(activation, max_activation)),
  }));
}
