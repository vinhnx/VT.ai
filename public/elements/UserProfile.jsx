import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function UserProfile() {
	return (
		<Card className="w-full max-w-md mx-auto mt-6">
			<CardHeader className="flex flex-col items-center gap-2">
                <Avatar>
                  <AvatarImage src={props.avatar_url || '/public/avatars/assistant.jpg'} alt="Avatar"/>
                  <AvatarFallback>VT</AvatarFallback>
                </Avatar>

				<CardTitle className="text-xl">{props.full_name || 'Anonymous'}</CardTitle>
				<div className="text-sm text-muted-foreground">{props.email}</div>
				<Badge variant="outline">{props.subscription_tier || 'free'}</Badge>
			</CardHeader>
			<CardContent>
				<div className="flex flex-col gap-2">
					<div><b>User ID:</b> {props.user_id}</div>
					<div><b>Provider:</b> {props.provider}</div>
					<div><b>Tokens Used:</b> {props.tokens_used}</div>
					<div><b>Joined:</b> {props.created_at?.slice(0, 10)}</div>
				</div>
			</CardContent>
		</Card>
	)
}
