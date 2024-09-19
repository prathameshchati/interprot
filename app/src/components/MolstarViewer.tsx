// Mostly AI generated code
// generated from this example https://embed.plnkr.co/plunk/JsL8TzofFtKq0ZV4

import { useEffect } from "react";

const MolstarViewer = () => {
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
          url: "https://alphafold.ebi.ac.uk/files/AF-A0PK11-F1-model_v4.cif",
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

      const viewerContainer = document.getElementById("myViewer");
      viewerInstance.render(viewerContainer, options);

      // Color resides in the viewer
      // https://github.com/molstar/pdbe-molstar/issues/90#issuecomment-2317239229
      // Needs to be wrapped in a setTimeout to ensure the viewer is ready
      setTimeout(() => {
        viewerInstance.visual.select({
          data: [
            { struct_asym_id: "A", residue_number: 1, color: "#ff0000" },
            { struct_asym_id: "A", residue_number: 2, color: "#ff8800" },
            {
              struct_asym_id: "A",
              start_residue_number: 3,
              end_residue_number: 6,
              color: "#ffff00",
            },
            {
              struct_asym_id: "A",
              start_residue_number: 15,
              end_residue_number: 20,
              color: "#88ff00",
            },
          ],
          nonSelectedColor: "#ffffff",
        });
      }, 500);
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
        id="myViewer"
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
