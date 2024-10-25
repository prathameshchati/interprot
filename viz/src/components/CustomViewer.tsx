import { useState, useEffect, useRef } from "react";
import { residueColor } from "../utils";
import SeqViewer from "./SeqViewer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { sequenceToTokens } from "../utils";
interface CustomViewerProps {
  feature: number;
}

const CustomViewer = ({ feature }: CustomViewerProps) => {
  const [activationList, setActivationList] = useState<number[]>([]);
  const [sequence, setSequence] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [showStructure, setShowStructure] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");
  const sequenceRef = useRef<string>("");

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
            sequence: sequence.toUpperCase(),
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
    sequenceRef.current = sequence;
    await fetchSAEActivations();
    setMessage("");
  };

  useEffect(() => {
    const renderStructure = async () => {
      if (!sequenceRef.current) return;
      setIsLoading(true);
      try {
        const response = await fetch("https://api.esmatlas.com/foldSequence/v1/pdb/", {
          method: "POST",
          headers: {
            "Content-Type": "text/plain",
          },
          body: sequenceRef.current.toUpperCase(),
        });

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        console.log("response", response);

        const pdbData = await response.text();
        const blob = new Blob([pdbData], { type: "text/plain" });
        const blobUrl = URL.createObjectURL(blob);

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

        viewerInstance.events.loadComplete.subscribe(() => {
          viewerInstance.visual.select({
            data: residueColor(activationList),
            nonSelectedColor: "#ffffff",
          });
          setMessage("Structure generated with ESMFold.");
        });
      } catch (error) {
        console.error("Error folding sequence:", error);
      } finally {
        setIsLoading(false);
      }
    };

    if (activationList.length === 0) {
      setShowStructure(false);
      return;
    }
    if (activationList.every((act) => act === 0)) {
      setMessage(
        "This feature did not activate on your sequence. Try a sequence more similar to ones below."
      );
      setShowStructure(false);
      return;
    }
    if (sequenceRef.current.length > 400) {
      setMessage(
        "No structure generated. We are folding with ESMFold API which has a limit of 400 residues. Please try a shorter sequence."
      );
      setShowStructure(false);
      return;
    }

    setShowStructure(true);
    renderStructure();
  }, [feature, activationList]);

  // Reset custom viewer state whenever user navigates to a new feature
  useEffect(() => {
    setActivationList([]);
    setSequence("");
    setIsLoading(false);
    setMessage("");
  }, [feature]);

  return (
    <div>
      <div style={{ marginTop: 20 }}>
        <div className="flex overflow-x-auto">
          <Input
            type="text"
            style={{ marginRight: 10 }}
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Enter your own protein sequence"
          />
          <Button onClick={handleSubmit} disabled={isLoading || !sequence}>
            {isLoading ? "Loading..." : "Submit"}
          </Button>
        </div>
      </div>
      {activationList.length > 0 && (
        <div style={{ margin: 20 }}>
          <SeqViewer
            seq={{
              tokens_acts_list: activationList,
              tokens_list: sequenceToTokens(sequence.toUpperCase()),
            }}
          />
        </div>
      )}
      {showStructure && (
        <div
          id="custom-viewer"
          style={{
            marginTop: 20,
            width: "100%",
            height: activationList.length > 0 ? 400 : 0,
            position: "relative",
          }}
        />
      )}
      {message && <small>{message}</small>}
    </div>
  );
};

export default CustomViewer;
