import { useEffect, useState, useRef } from "react";
import { residueColor } from "../utils";
import proteinEmoji from "../protein.png";
import { useIsMobile } from "@/hooks/use-mobile";

interface ProteinData {
  alphafold_id: string;
  tokens_acts_list: Array<number>;
}

interface MolstarViewerProps {
  proteins: ProteinData[];
}

const MOBILE_RENDER_LIMIT = 3;
const RENDER_DELAY = 150;

const MolstarViewer = ({ proteins }: MolstarViewerProps) => {
  const [previewImages, setPreviewImages] = useState<string[]>([]);
  const [activeViewers, setActiveViewers] = useState<Set<number>>(new Set());
  const viewerInstancesRef = useRef<Map<number, any>>(new Map());
  const offscreenContainerRef = useRef<HTMLDivElement | null>(null);
  const offscreenViewerRef = useRef<any>(null);
  const isMobile = useIsMobile();
  const mountedRef = useRef(true);

  const initializeOffscreenViewer = () => {
    if (!offscreenContainerRef.current) {
      const container = document.createElement("div");
      container.style.width = "400px";
      container.style.height = "400px";
      container.style.position = "absolute";
      container.style.left = "-9999px";
      document.body.appendChild(container);
      offscreenContainerRef.current = container;

      if (offscreenViewerRef.current?.destroy) {
        offscreenViewerRef.current.destroy();
      }
      // @ts-expect-error
      const viewer = new PDBeMolstarPlugin();
      offscreenViewerRef.current = viewer;
    }
  };

  const renderToImage = async (protein: ProteinData) => {
    return new Promise<string>((resolve, reject) => {
      if (!mountedRef.current) {
        reject(new Error('Component unmounted'));
        return;
      }

      if (!offscreenContainerRef.current || !offscreenViewerRef.current) {
        initializeOffscreenViewer();
      }

      const viewer = offscreenViewerRef.current;
      const options = {
        customData: {
          url: `https://alphafold.ebi.ac.uk/files/AF-${protein.alphafold_id}-F1-model_v4.cif`,
          format: "cif",
        },
        alphafoldView: true,
        bgColor: { r: 255, g: 255, b: 255 },
        hideControls: true,
        hideCanvasControls: ["selection", "animation", "controlToggle", "controlInfo"],
        landscape: true,
      };

      const timeoutId = setTimeout(() => {
        reject(new Error('Render timeout'));
      }, 10000); // 10 second timeout

      try {
        viewer.render(offscreenContainerRef.current, options);
        viewer.events.loadComplete.subscribe(() => {
          if (!mountedRef.current) {
            clearTimeout(timeoutId);
            reject(new Error('Component unmounted'));
            return;
          }

          viewer.visual.select({
            data: residueColor(protein.tokens_acts_list),
            nonSelectedColor: "#ffffff",
          });

          setTimeout(() => {
            const canvas = offscreenContainerRef.current?.querySelector("canvas");
            if (canvas) {
              const image = canvas.toDataURL("image/png");
              const ctx = canvas.getContext("2d");
              ctx?.clearRect(0, 0, canvas.width, canvas.height);
              clearTimeout(timeoutId);
              resolve(image);
            }
          }, RENDER_DELAY);
        });
      } catch (error) {
        clearTimeout(timeoutId);
        reject(error);
      }
    });
  };

  const renderSequentially = async (proteins: ProteinData[]) => {
    const images: string[] = [];
    const renderSet = isMobile ? proteins.slice(0, MOBILE_RENDER_LIMIT) : proteins;
    
    for (const protein of renderSet) {
      if (!mountedRef.current) break;
      
      try {
        const image = await renderToImage(protein);
        if (mountedRef.current) {
          images.push(image);
          setPreviewImages([...images]);
        }
      } catch (error) {
        console.error('Failed to render protein:', protein.alphafold_id, error);
        images.push(''); // Push empty string to maintain index alignment
      }
    }
    return images;
  };

  const loadMolstarPlugin = (protein: ProteinData, index: number) => {
    // Cleanup existing viewer if it exists
    if (viewerInstancesRef.current.has(index)) {
      const existingViewer = viewerInstancesRef.current.get(index);
      if (existingViewer?.destroy) {
        existingViewer.destroy();
      }
      viewerInstancesRef.current.delete(index);
    }

    // Create new viewer
    // @ts-expect-error
    const viewer = new PDBeMolstarPlugin();
    viewerInstancesRef.current.set(index, viewer);

    const options = {
      customData: {
        url: `https://alphafold.ebi.ac.uk/files/AF-${protein.alphafold_id}-F1-model_v4.cif`,
        format: "cif",
      },
      alphafoldView: true,
      bgColor: { r: 255, g: 255, b: 255 },
      hideControls: true,
      hideCanvasControls: ["selection", "animation", "controlToggle", "controlInfo"],
      sequencePanel: true,
      landscape: true,
    };

    const viewerContainer = document.getElementById(`viewer-${index}`);
    if (viewerContainer && mountedRef.current) {
      viewer.render(viewerContainer, options);
      viewer.events.loadComplete.subscribe(() => {
        if (mountedRef.current) {
          viewer.visual.select({
            data: residueColor(protein.tokens_acts_list),
            nonSelectedColor: "#ffffff",
          });
        }
      });
    }
  };

  const cleanup = () => {
    // Clean up all active viewers
    viewerInstancesRef.current.forEach((viewer) => {
      if (viewer?.destroy) {
        viewer.destroy();
      }
    });
    viewerInstancesRef.current.clear();

    // Clean up offscreen viewer
    if (offscreenViewerRef.current?.destroy) {
      offscreenViewerRef.current.destroy();
    }
    offscreenViewerRef.current = null;

    // Remove offscreen container
    if (offscreenContainerRef.current) {
      document.body.removeChild(offscreenContainerRef.current);
      offscreenContainerRef.current = null;
    }

    // Reset state
    setActiveViewers(new Set());
    setPreviewImages([]);
  };

  useEffect(() => {
    mountedRef.current = true;
    const scriptId = "molstar-script";
    let script = document.getElementById(scriptId);

    // Clean up previous state when proteins change
    cleanup();

    const initializeViewer = async () => {
      if (mountedRef.current) {
        await renderSequentially(proteins);
      }
    };

    if (!script) {
      script = document.createElement("script");
      script.id = scriptId;
      // @ts-expect-error
      script.src = "https://cdn.jsdelivr.net/npm/pdbe-molstar@3.3.0/build/pdbe-molstar-plugin.js";
      script.onload = initializeViewer;
      document.body.appendChild(script);
    } else {
      initializeViewer();
    }

    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [proteins]);

  const handleImageClick = (index: number) => {
    if (isMobile && activeViewers.size >= 3) {
      // Limit active viewers to 3 at a time
      const oldestViewer = Array.from(activeViewers)[0];
      setActiveViewers((prev) => {
        const newSet = new Set(prev);
        newSet.delete(oldestViewer);
        const existingViewer = viewerInstancesRef.current.get(oldestViewer);
        if (existingViewer?.destroy) {
          existingViewer.destroy();
        }
        viewerInstancesRef.current.delete(oldestViewer);
        return newSet;
      });
    }

    setActiveViewers((prev) => {
      const newSet = new Set(prev);
      if (!newSet.has(index)) {
        newSet.add(index);
        setTimeout(() => loadMolstarPlugin(proteins[index], index), 0);
      }
      return newSet;
    });
  };

  return (
    <div className="container mx-auto p-4">
      <div className="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {proteins.slice(0, isMobile ? MOBILE_RENDER_LIMIT : proteins.length).map((protein, index) => (
          <div key={protein.alphafold_id} className="relative aspect-square">
            {activeViewers.has(index) ? (
              <div
                id={`viewer-${index}`}
                className="w-full h-full"
                onClick={() => handleImageClick(index)}
              />
            ) : previewImages[index] ? (
              <img
                src={previewImages[index]}
                alt={`Protein ${protein.alphafold_id}`}
                onClick={() => handleImageClick(index)}
                className="w-full h-full object-cover cursor-pointer rounded-lg hover:opacity-80 transition-opacity"
              />
            ) : (
              <div className="flex flex-col items-center justify-center w-full h-full">
                <img
                  src={proteinEmoji}
                  alt="Loading..."
                  className="w-12 h-12 animate-wiggle mb-4"
                />
              </div>
            )}
            <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
              {protein.alphafold_id}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MolstarViewer;