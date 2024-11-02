export type CuratedFeature = {
  name: string;
  dim: number;
  desc: string;
  group?: string;
};

export type SAEConfig = {
  baseUrl: string;
  numHiddenDims: number;
  plmLayer: number;
  curated?: CuratedFeature[];
  defaultDim: number;
  supportsCustomSequence?: boolean;
};

export const SAE_CONFIGS: Record<string, SAEConfig> = {
  "SAE4096-L24": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l24_sae4096_100Kseqs/",
    numHiddenDims: 4096,
    plmLayer: 24,
    curated: [
      {
        name: "full turn in alpha helix",
        dim: 2293,
        desc: "This feature activates strongly on every 7th amino acid (approximately two full turns) in alpha helices.",
        group: "structural",
      },
      {
        name: "beta strand channel",
        dim: 3883,
        desc: "This feature activates on channel-like structures consisting of beta strands. It fires on a specific beta hairpin within the channel.",
        group: "structural",
      },
      {
        name: "hugging helices",
        dim: 3348,
        desc: "This feature activates on the interfacing residues of bunched-together alpha helices. It seems to understand the orientation of the helices as well as the surface exposed to the opposing helices, firing on either a single amino acid or 2 adjacent ones depending on the surface.",
        group: "structural",
      },
      {
        name: "kinase helix",
        dim: 594,
        desc: "This feature activates strongly on a specific helix in kinase domains and weakly on surrounding beta strands. The highlighted helix is always the one opposed to the beta sheet.",
        group: "structural",
      },
      {
        name: "kinase beta mids",
        dim: 294,
        desc: "Activates on the middle residues in the beta sheet in kinase domains. It fires most strongly on the second outer beta strand and doesn't fire on the outermost beta strand at all.",
        group: "structural",
      },
      {
        name: "long helix every turn",
        dim: 214,
        desc: "This feature activates on interspersed amino acids in long helices in the pattern 100100010010001..., which corresponds to every turn.",
        group: "structural",
      },
      {
        name: "beta strand motif",
        dim: 88,
        desc: "This feature activates on a specific beta strand motif in ABC transporters. The highlighted beta strand is always the one that opposes an alpha helix.",
        group: "structural",
      },
      {
        name: "perpendicular helix",
        dim: 2320,
        desc: "This feature activates on parts of alpha helices. The highlighted parts tend to be perpendicular to a nearby helix.",
        group: "structural",
      },
      {
        name: "single beta strand",
        dim: 1299,
        desc: "Activates on a single beta strand",
        group: "structural",
      },
      {
        name: "alpha helix: first aa",
        dim: 3451,
        desc: "Activates on the first amino acid in a specific alpha helix",
        group: "structural",
      },
      {
        name: "beta strand: first aa",
        dim: 782,
        desc: "Activates on the first amino acid in a specific beta strand",
        group: "structural",
      },
      {
        name: "beta helix",
        dim: 250,
        desc: "Activates on short beta strands in beta helices",
        group: "structural",
      },
      {
        name: "alpha helix turn",
        dim: 56,
        desc: "Activates on the turn between two alpha helices in ABC transporter proteins",
        group: "structural",
      },
      {
        name: "long alpha helices",
        dim: 1008,
        desc: "Activates on most amino acids in long alpha helices",
        group: "structural",
      },
      {
        name: "disordered",
        dim: 2763,
        desc: "Activates on disordered regions containing K, A, and P residues",
        group: "structural",
      },
      {
        name: "leucine rich repeats",
        dim: 3425,
        desc: "Activates on the amino acid before the start of a beta strand in a leucine rich repeat",
        group: "structural",
      },
      {
        name: "helix bunch",
        dim: 3055,
        desc: "Activates on a bunch of alpha helices",
        group: "structural",
      },
      {
        name: "start M",
        dim: 600,
        desc: "Activates on the M amino acid at the start of a sequence",
        group: "amino acid position",
      },
      {
        name: "second residue",
        dim: 3728,
        desc: "Mostly activates on the second amino acid in a sequence",
        group: "amino acid position",
      },
      {
        name: "alanine",
        dim: 3267,
        desc: "Activates on alanine residues",
        group: "amino acid identity",
      },
      {
        name: "aspartic acid",
        dim: 2830,
        desc: "Activates on aspartic acid residues",
        group: "amino acid identity",
      },
      {
        name: "glutamic acid",
        dim: 2152,
        desc: "Activates on glutamic acid residues",
        group: "amino acid identity",
      },
      {
        name: "phenylalanine",
        dim: 252,
        desc: "Activates on phenylalanine residues",
        group: "amino acid identity",
      },
      {
        name: "aspartic acid",
        dim: 3830,
        desc: "Activates on aspartic acid residues",
        group: "amino acid identity",
      },
      {
        name: "histidine",
        dim: 743,
        desc: "Activates on histidine residues",
        group: "amino acid identity",
      },
      {
        name: "isoleucine",
        dim: 3978,
        desc: "Activates on isoleucine residues",
        group: "amino acid identity",
      },
      {
        name: "lysine",
        dim: 3073,
        desc: "Activates on lysine residues",
        group: "amino acid identity",
      },
      {
        name: "leucine",
        dim: 1497,
        desc: "Activates on leucine residues",
        group: "amino acid identity",
      },
      {
        name: "methionine",
        dim: 444,
        desc: "Activates on methionine residues",
        group: "amino acid identity",
      },
      {
        name: "asparagine",
        dim: 21,
        desc: "Activates on asparagine residues",
        group: "amino acid identity",
      },
      {
        name: "proline",
        dim: 1386,
        desc: "Activates on proline residues",
        group: "amino acid identity",
      },
      {
        name: "glutamine",
        dim: 1266,
        desc: "Activates on glutamine residues",
        group: "amino acid identity",
      },
      {
        name: "tryptophan",
        dim: 2685,
        desc: "Activates on tryptophan residues",
        group: "amino acid identity",
      },
      {
        name: "tyrosine",
        dim: 3481,
        desc: "Activates on tyrosine residues",
        group: "amino acid identity",
      },
      {
        name: "kinase beta strands",
        dim: 3642,
        desc: "Activates on some beta strands in kinase domains and weakly on the C-helix",
        group: "structural",
      },
      {
        name: "kinase beta strand",
        dim: 3260,
        desc: "Activates on a beta strand in kinase domains",
        group: "structural",
      },
      {
        name: "kinase beta strand",
        dim: 16,
        desc: "Activates on a beta strand in kinase domains",
        group: "structural",
      },
      {
        name: "beta strand hammock",
        dim: 179,
        desc: "Activates on a beta strand and the disordered regions at each end",
        group: "structural",
      },
    ],
    defaultDim: 2293,
    supportsCustomSequence: true,
  },
  "SAE16384-L5": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l5_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 5,
    defaultDim: 0,
  },
  "SAE16384-L10": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l10_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 10,
    defaultDim: 0,
  },
  "SAE16384-L15": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l15_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 15,
    defaultDim: 0,
  },
  "SAE16384-L20": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l20_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 20,
    defaultDim: 0,
  },
  "SAE16384-L25": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l25_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 25,
    defaultDim: 0,
  },
  "SAE16384-L30": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l20_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 30,
    defaultDim: 0,
  },
  "SAE16384-L33": {
    baseUrl:
      "https://raw.githubusercontent.com/liambai/plm-interp-viz-data/refs/heads/main/esm2_plm1280_l20_sae16384_1Mseqs/",
    numHiddenDims: 16384,
    plmLayer: 33,
    defaultDim: 0,
  },
};
