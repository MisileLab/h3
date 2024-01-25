import { useAuth0 } from "@auth0/auth0-react";
import { useEffect, useState } from "react";

const domain = "the-simple-site.jp.auth0.com";

function App() {
  const auth0 = useAuth0();
  const [token, setToken] = useState("");
  useEffect(()=>{
    if (!auth0.isAuthenticated) {return;}
    const a = async () => {
      try {
        setToken(await auth0.getAccessTokenSilently({
          authorizationParams: {
            audience: `https://${domain}/api/v2/`,
            scope: "read:current_user"
          },
        }));
      } catch (e) {
        console.error(e);
        try {
          const confirm = await auth0.getAccessTokenWithPopup({
            authorizationParams: {
              audience: `https://${domain}/api/v2/`,
              scope: "read:current_user"
            }
          });
          if (confirm !== undefined) {setToken(confirm)}
        } catch (e) {console.error(e);}
      }
    }
    a();
  })
  return (
    <div>
      <h1>simple site with React</h1>
      <h2>Login with some <button onClick={()=>auth0.loginWithRedirect()}>button</button></h2>
      {auth0.isAuthenticated && <div>
        <p>Your ID: {auth0.user?.name}</p>
        <p>Your Secret ID: {auth0.user?.email}</p>
        <p>Your Token: {token}</p>
        <button onClick={()=>auth0.logout({logoutParams: {returnTo: window.location.origin}})}>logout</button>
      </div>}
    </div>
  );
}

export default App;
