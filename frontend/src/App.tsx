import { BrowserRouter } from 'react-router-dom';
import { Shell } from './components/layout/Shell';
import { AppRoutes } from './routes';

export default function App() {
  return (
    <BrowserRouter>
      <Shell>
        <AppRoutes />
      </Shell>
    </BrowserRouter>
  );
}
