import { Link } from "react-router-dom";

const HomeNavigator: React.FC = () => {
  return (
    <Link
      to="/"
      className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-0 flex items-center gap-3 justify-center h-20"
    >
      <img src="/logo.png" alt="InterProt Logo" className="h-11 w-11" />
      InterProt
    </Link>
  );
};

export default HomeNavigator;
