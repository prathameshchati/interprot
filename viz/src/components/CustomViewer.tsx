import { useState, useEffect } from "react";
import { residueColor } from "../utils";

interface CustomViewerProps {
  feature: number;
}

const CustomViewer = ({ feature }: CustomViewerProps) => {
  const [activationList, setActivationList] = useState<number[]>([]);
  const [sequence, setSequence] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const fetchSAEActivations = async () => {
    if (!sequence) return;
    setIsLoading(true);
    try {
      const response = await fetch("https://api.runpod.ai/v2/yk9ehzl3h653vj/runsync", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${import.meta.env.VITE_RUNPOD_API_KEY}`,
        },
        body: JSON.stringify({
          input: {
            sequence: sequence,
            dim: feature,
          },
        }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const resp = await response.json();
      const data = resp.output.data;
      if (data.tokens_acts_list) {
        setActivationList(data.tokens_acts_list);
      } else {
        console.error("Unexpected data format:", data);
      }
    } catch (error) {
      console.error("Error fetching activation data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    await fetchSAEActivations();
  };

  useEffect(() => {
    const showStructure = async () => {
      if (!sequence) return;
      setIsLoading(true);
      try {
        const response = await fetch("https://api.esmatlas.com/foldSequence/v1/pdb/", {
          method: "POST",
          headers: {
            "Content-Type": "text/plain",
          },
          body: sequence,
        });

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        const pdbData = await response.text();

        // Create a Blob with the PDB data
        const blob = new Blob([pdbData], { type: "text/plain" });
        const blobUrl = URL.createObjectURL(blob);

        // Update the viewer with the new PDB data
        // @ts-expect-error
        const viewerInstance = new PDBeMolstarPlugin();
        const options = {
          customData: {
            url: blobUrl,
            format: "pdb",
          },
          alphafoldView: false,
          bgColor: { r: 255, g: 255, b: 255 },
          hideControls: true,
          hideCanvasControls: ["selection", "animation", "controlToggle", "controlInfo"],
          sequencePanel: true,
          landscape: true,
        };

        const viewerContainer = document.getElementById("custom-viewer");
        viewerInstance.render(viewerContainer, options);

        // Listen for the 'load' event
        viewerInstance.events.loadComplete.subscribe(() => {
          viewerInstance.visual.select({
            data: residueColor(activationList),
            nonSelectedColor: "#ffffff",
          });
        });
      } catch (error) {
        console.error("Error folding sequence:", error);
      } finally {
        setIsLoading(false);
      }
    };

    if (activationList.length > 0) {
      showStructure();
    }
  }, [feature, sequence, activationList]);

  useEffect(() => {
    setActivationList([]);
    setSequence("");
    setIsLoading(false);
  }, [feature]);

  return (
    <div>
      <div style={{ marginTop: 20 }}>
        <div className="flex overflow-x-auto">
          <input
            type="text"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Enter protein sequence"
            style={{
              width: "100%",
              padding: "10px",
              fontSize: "16px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              marginRight: "10px",
              outline: "none",
            }}
          />
          <button
            onClick={handleSubmit}
            disabled={isLoading || !sequence}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              backgroundColor: isLoading || !sequence ? "#ccc" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: isLoading || !sequence ? "not-allowed" : "pointer",
              transition: "background-color 0.3s",
            }}
          >
            {isLoading ? "Loading..." : "Analyze"}
          </button>
        </div>
      </div>
      <div
        id="custom-viewer"
        style={{
          marginTop: 20,
          width: "100%",
          height: activationList.length > 0 ? 400 : 0,
          position: "relative",
        }}
      />
    </div>
  );
};

export default CustomViewer;
