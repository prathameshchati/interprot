export type CuratedFeature = {
  name: string;
  dim: number;
  desc: string;
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
        name: "free alpha helix alternating",
        dim: 2293,
        desc: "This feature activates on interspersed amino acids in alpha helices that tend to be isolated from other structures. It fires strongly in every other turn of the helix (and weakly on the other turns).",
      },
      {
        name: "beta strand channel",
        dim: 3883,
        desc: "This feature activates on channel-like structures consisting of beta strands. It fires only on a subset of beta strands consisting of the channel and fires more strongly one side of the channel.",
      },
      {
        name: "hugging helices",
        dim: 3348,
        desc: "This feature activates on the interfacing residues of bunched-together alpha helices. In addition to understanding the geometric orientation of the helices, it seems to also capture the concept of the binding interface because it fires on either a single amino acid or 2 adjacent amino acids depending on the surface exposed to the opposing helix.",
      },
      {
        name: "long helix every turn",
        dim: 214,
        desc: "This feature activates on interspersed amino acids in long helices in the pattern 100100010010001..., which corresponds to every turn.",
      },
      {
        name: "beta strand motif",
        dim: 88,
        desc: "This feature activates on a specific beta strand motif in ABC transporters. The highlighted beta strand is always the one that opposes an alpha helix.",
      },
      {
        name: "single beta strand",
        dim: 1299,
        desc: "Activates on a single beta strand",
      },
      {
        name: "helix: first aa",
        dim: 3451,
        desc: "Activates on the first amino acid in a specific alpha helix",
      },
      {
        name: "beta strand: first aa",
        dim: 782,
        desc: "Activates on the first amino acid in a specific beta strand",
      },
      {
        name: "beta helix",
        dim: 250,
        desc: "Activates on short beta strands in beta helices",
      },
      {
        name: "alpha helix turn",
        dim: 56,
        desc: "Activates on the turn between two alpha helices in ABC transporter proteins",
      },
      {
        name: "long alpha helices",
        dim: 1008,
        desc: "Activates on most amino acids in long alpha helices",
      },
      {
        name: "disordered",
        dim: 2763,
        desc: "Activates on disordered regions containing K, A, and P residues",
      },
      {
        name: "leucine rich repeats",
        dim: 3425,
        desc: "Activates on the amino acid before the start of a beta strand in a leucine rich repeat",
      },
      {
        name: "helix bunch",
        dim: 3055,
        desc: "Activates on a bunch of alpha helices",
      },
      {
        name: "start M",
        dim: 600,
        desc: "Activates on the M amino acid at the start of a sequence",
      },
      {
        name: "second residue",
        dim: 3728,
        desc: "Mostly activates on the second amino acid in a sequence",
      },
      { name: "alanine", dim: 3267, desc: "Activates on alanine residues" },
      {
        name: "aspartic acid",
        dim: 2830,
        desc: "Activates on aspartic acid residues",
      },
      {
        name: "glutamic acid",
        dim: 2152,
        desc: "Activates on glutamic acid residues",
      },
      {
        name: "phenylalanine",
        dim: 252,
        desc: "Activates on phenylalanine residues",
      },
      {
        name: "aspartic acid",
        dim: 3830,
        desc: "Activates on aspartic acid residues",
      },
      { name: "histidine", dim: 743, desc: "Activates on histidine residues" },
      {
        name: "isoleucine",
        dim: 3978,
        desc: "Activates on isoleucine residues",
      },
      { name: "lysine", dim: 3073, desc: "Activates on lysine residues" },
      { name: "leucine", dim: 1497, desc: "Activates on leucine residues" },
      {
        name: "methionine",
        dim: 444,
        desc: "Activates on methionine residues",
      },
      { name: "asparagine", dim: 21, desc: "Activates on asparagine residues" },
      { name: "proline", dim: 1386, desc: "Activates on proline residues" },
      { name: "glutamine", dim: 1266, desc: "Activates on glutamine residues" },
      { name: "arginine", dim: 3569, desc: "Activates on arginine residues" },
      { name: "serine", dim: 1473, desc: "Activates on serine residues" },
      { name: "threonine", dim: 220, desc: "Activates on threonine residues" },
      { name: "valine", dim: 3383, desc: "Activates on valine residues" },
      {
        name: "tryptophan",
        dim: 2685,
        desc: "Activates on tryptophan residues",
      },
      { name: "tyrosine", dim: 3481, desc: "Activates on tyrosine residues" },
      {
        name: "kinase helix",
        dim: 594,
        desc: "Activates strongly on the C-helix in kinase domains and weakly on surrounding beta strands",
      },
      {
        name: "kinase beta strands",
        dim: 3642,
        desc: "Activates on some beta strands in kinase domains and weakly on the C-helix",
      },
      {
        name: "middle residues in kinase beta strands",
        dim: 294,
        desc: "Activates on the middle residues in kinase domain beta strands",
      },
      {
        name: "kinase beta strand",
        dim: 3260,
        desc: "Activates on a beta strand in kinase domains",
      },
      {
        name: "kinase beta strand",
        dim: 16,
        desc: "Activates on a beta strand in kinase domains",
      },
      {
        name: "beta strand hammock",
        dim: 179,
        desc: "Activates on a beta strand and the disordered regions at each end",
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
