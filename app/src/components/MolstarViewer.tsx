// Mostly AI generated code
// generated from this example https://embed.plnkr.co/plunk/JsL8TzofFtKq0ZV4

import { useEffect } from "react";
import { redColorMap } from "./SeqViewer";
interface MolstarViewerProps {
  alphafold_id: string;
  activation_list: Array<number>;
}

function rgbToHex(rgb: string): string {
  const [r, g, b] = rgb.match(/\d+/g)!.map(Number);
  const hex = ((r << 16) | (g << 8) | b).toString(16).padStart(6, "0");
  return `#${hex}`;
}

function residueColor(activation_list: Array<number>) {
  const max_activation = Math.max(...activation_list);
  console.warn("max_activation", max_activation);

  return activation_list.map((activation, i) => ({
    struct_asym_id: "A",
    residue_number: i + 1,
    color: rgbToHex(redColorMap(activation, max_activation)),
  }));
}

const MolstarViewer = ({
  alphafold_id,
  activation_list,
}: MolstarViewerProps) => {
  useEffect(() => {
    // Dynamically load the Molstar script
    const script = document.createElement("script");
    script.src =
      "https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js";
    script.onload = () => {
      // Create plugin instance and set options after script loads
      // @ts-expect-error
      const viewerInstance = new PDBeMolstarPlugin();

      const options = {
        customData: {
          url: `https://alphafold.ebi.ac.uk/files/AF-${alphafold_id}-F1-model_v4.cif`,
          format: "cif",
        },
        alphafoldView: true,
        bgColor: { r: 255, g: 255, b: 255 },
        hideControls: true, // Hide all controls
        hideCanvasControls: [
          "selection",
          "animation",
          "controlToggle",
          "controlInfo",
        ],
        sequencePanel: true,
        landscape: true,
      };

      const viewerContainer = document.getElementById(`viewer-${alphafold_id}`);
      viewerInstance.render(viewerContainer, options);

      // Color resides in the viewer
      // https://github.com/molstar/pdbe-molstar/issues/90#issuecomment-2317239229
      // Needs to be wrapped in a setTimeout to ensure the viewer is ready
      setTimeout(() => {
        console.warn(residueColor(activation_list));
        viewerInstance.visual.select({
          data: residueColor(activation_list),
          nonSelectedColor: "#ffffff",
        });
      }, 5000);
    };
    document.body.appendChild(script);

    // Cleanup script on unmount
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div>
      <h3>PDBe Mol* JS Plugin</h3>
      <div
        id={`viewer-${alphafold_id}`}
        style={{
          float: "left",
          width: "700px",
          height: "400px",
          position: "relative",
          margin: "20px",
        }}
      ></div>
    </div>
  );
};

export default MolstarViewer;
